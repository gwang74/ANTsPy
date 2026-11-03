"""
Friendly Python wrapper for the CUDA registration entry point.
"""

__all__ = ["gpu_registration", "apply_transform_map"]

import numpy as np

from ants.internal import get_lib_fn


_DIRECTION_TO_OUTPUT = {
    "moving_to_fixed": 0,
    "fixed_to_moving": 1,
    "template_to_moving": 1,
}

_PRESETS = {
    "defect": {
        "affine_iterations": (2100, 1200, 1200, 0),
        "syn_iterations": (40, 20, 0),
        "robust_threshold": 0.25,
        "syn_max_step": 0.2,
        "syn_max_displacement": 6.0,
    },
    "fast": {
        "affine_iterations": (1200, 800, 400, 0),
        "syn_iterations": (30, 15, 0),
        "robust_threshold": 0.35,
        "syn_max_step": 0.16,
        "syn_max_displacement": 6.0,
    },
}


def apply_transform_map(
    image,
    transform_map,
    interpolation="linear",
    border_value=0,
):
    """Warp a derived image with a map returned by ``gpu_registration``.

    The source image must share the geometry of the fixed image used during
    registration. Grayscale and multi-channel NumPy/OpenCV images are accepted.
    Use nearest-neighbor interpolation for labels, masks, and categorical data.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("apply_transform_map requires OpenCV (cv2)") from exc

    source = np.asarray(image)
    sampling_map = np.asarray(transform_map, dtype=np.float32)
    if source.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D multi-channel array")
    if sampling_map.ndim != 3 or sampling_map.shape[2] != 2:
        raise ValueError("transform_map must have shape (height, width, 2)")
    if source.shape[:2] != sampling_map.shape[:2]:
        raise ValueError(
            "image spatial shape must match the fixed image geometry used "
            "to create transform_map"
        )

    interpolation_key = str(interpolation).lower()
    interpolation_flags = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
    }
    if interpolation_key not in interpolation_flags:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    # The returned map is already an absolute output-to-source coordinate map,
    # so no affine or displacement composition is needed at application time.
    return cv2.remap(
        source,
        sampling_map[..., 0],
        sampling_map[..., 1],
        interpolation=interpolation_flags[interpolation_key],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _normalize_direction(direction):
    if isinstance(direction, str):
        key = direction.lower()
        if key not in _DIRECTION_TO_OUTPUT:
            raise ValueError(
                "direction must be 'fixed_to_moving', 'template_to_moving', "
                "or 'moving_to_fixed'"
            )
        return _DIRECTION_TO_OUTPUT[key]

    if direction in (0, 1):
        return int(direction)

    raise ValueError(
        "direction must be 'fixed_to_moving', 'template_to_moving', "
        "'moving_to_fixed', 0, or 1"
    )


def _as_registration_array(image, name):
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("%s must be a 2D grayscale image array" % name)
    if array.size == 0:
        raise ValueError("%s must not be empty" % name)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("%s must contain numeric pixel values" % name)
    return array


def _uniform_blur_axis(image, radius, axis):
    if radius <= 0:
        return image

    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="edge")
    cumsum = np.cumsum(padded, axis=axis, dtype=np.float32)

    zero_shape = list(cumsum.shape)
    zero_shape[axis] = 1
    cumsum = np.concatenate([np.zeros(zero_shape, dtype=np.float32), cumsum], axis=axis)

    length = image.shape[axis]
    window = np.float32(radius * 2 + 1)
    end = np.arange(window, window + length)
    start = np.arange(0, length)
    return (np.take(cumsum, end, axis=axis) - np.take(cumsum, start, axis=axis)) / window


def _robust_rescale(image):
    lower, upper = np.percentile(image, (1.0, 99.0))
    if upper <= lower:
        lower = float(np.min(image))
        upper = float(np.max(image))
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, lower, upper)
    return ((image - lower) * (255.0 / (upper - lower))).astype(np.float32)


def _background_subtract_metric(image, sigma):
    radius = max(1, int(round(float(sigma))))
    source = np.asarray(image, dtype=np.float32)

    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        window = radius * 2 + 1
        background = cv2.blur(source, (window, window), borderType=cv2.BORDER_REPLICATE)
        return _robust_rescale(source - background)

    # A separable box blur is intentionally used here: it is cheap, has no
    # external dependency, and is sufficient for removing slow background drift.
    background = _uniform_blur_axis(source, radius, axis=1)
    background = _uniform_blur_axis(background, radius, axis=0)
    return _robust_rescale(source - background)


def _local_normalize_metric(image, sigma):
    radius = max(1, int(round(float(sigma))))
    source = np.asarray(image, dtype=np.float32)

    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        window = radius * 2 + 1
        local_mean = cv2.blur(source, (window, window), borderType=cv2.BORDER_REPLICATE)
        local_square_mean = cv2.blur(source * source, (window, window), borderType=cv2.BORDER_REPLICATE)
        local_variance = np.maximum(local_square_mean - local_mean * local_mean, 1.0)
        return _robust_rescale((source - local_mean) / np.sqrt(local_variance))

    # This approximates a local NCC-style input without changing the CUDA SyN
    # metric itself: subtract local mean, divide by local contrast, then rescale.
    local_mean = _uniform_blur_axis(source, radius, axis=1)
    local_mean = _uniform_blur_axis(local_mean, radius, axis=0)
    local_square_mean = _uniform_blur_axis(source * source, radius, axis=1)
    local_square_mean = _uniform_blur_axis(local_square_mean, radius, axis=0)
    local_variance = np.maximum(local_square_mean - local_mean * local_mean, 1.0)
    return _robust_rescale((source - local_mean) / np.sqrt(local_variance))


def _choose_metric_preprocess(fixed, moving):
    fixed_source = np.asarray(fixed, dtype=np.float32)
    moving_source = np.asarray(moving, dtype=np.float32)

    diff = np.abs(fixed_source - moving_source)
    high_residual = float(np.percentile(diff, 95.0))

    # Separate slow illumination drift from high-frequency edge residuals.
    # Text and fine graphics can have a large raw P95 after a sub-pixel shift,
    # but their low-pass residual remains small.  Such pairs must not trigger
    # pre-warp local normalization because normalization and warping do not
    # commute around sharp edges.
    radius = max(4, int(round(min(fixed_source.shape) / 64.0)))
    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        window = radius * 2 + 1
        fixed_low = cv2.blur(
            fixed_source, (window, window), borderType=cv2.BORDER_REPLICATE
        )
        moving_low = cv2.blur(
            moving_source, (window, window), borderType=cv2.BORDER_REPLICATE
        )
    else:
        fixed_low = _uniform_blur_axis(fixed_source, radius, axis=1)
        fixed_low = _uniform_blur_axis(fixed_low, radius, axis=0)
        moving_low = _uniform_blur_axis(moving_source, radius, axis=1)
        moving_low = _uniform_blur_axis(moving_low, radius, axis=0)

    low_residual = float(np.percentile(np.abs(fixed_low - moving_low), 90.0))
    low_frequency_ratio = low_residual / max(high_residual, 1.0)

    def edge_density(image):
        gx = np.abs(np.diff(image, axis=1))
        gy = np.abs(np.diff(image, axis=0))
        return (float(np.mean(gx > 20.0)) + float(np.mean(gy > 20.0))) * 0.5

    edge_delta = abs(edge_density(fixed_source) - edge_density(moving_source))

    # Iterative LNCC handles local contrast changes inside the CUDA metric.  The
    # remaining optional preprocessing is reserved for genuine low-frequency
    # shading drift, not high-contrast text or defect residuals.
    if (
        high_residual >= 43.0
        and low_residual >= 15.0
        and low_frequency_ratio >= 0.45
        and edge_delta <= 0.006
    ):
        return "background_subtract"
    return None


def _make_metric_image(image, metric_preprocess, metric_sigma, name):
    if metric_preprocess is None or metric_preprocess == "none":
        return None

    method = str(metric_preprocess).lower()
    if method == "background_subtract":
        return _background_subtract_metric(image, metric_sigma)
    if method == "local_normalize":
        return _local_normalize_metric(image, metric_sigma)

    raise ValueError(
        "%s metric_preprocess must be None, 'none', 'background_subtract', "
        "'local_normalize', or 'auto'" % name
    )


def _resolve_preset(
    preset,
    affine_iterations,
    syn_iterations,
    robust_threshold,
    syn_max_step,
    syn_max_displacement,
):
    if preset is None:
        preset = "defect"
    preset_key = preset.lower()
    if preset_key not in _PRESETS:
        raise ValueError(
            "preset must be one of: %s" % ", ".join(sorted(_PRESETS))
        )

    values = dict(_PRESETS[preset_key])
    if affine_iterations is not None:
        values["affine_iterations"] = affine_iterations
    if syn_iterations is not None:
        values["syn_iterations"] = syn_iterations
    if robust_threshold is not None:
        values["robust_threshold"] = robust_threshold
    if syn_max_step is not None:
        values["syn_max_step"] = syn_max_step
    if syn_max_displacement is not None:
        values["syn_max_displacement"] = syn_max_displacement
    return values


def gpu_registration(
    fixed_image,
    moving_image,
    direction="fixed_to_moving",
    preset="defect",
    affine_iterations=None,
    syn_iterations=None,
    robust_threshold=None,
    syn_max_step=None,
    syn_max_displacement=None,
    metric_preprocess=None,
    metric_sigma=15.0,
    fixed_metric_image=None,
    moving_metric_image=None,
    output_dtype=np.uint8,
    profile=False,
    return_transform=False,
):
    """
    Register two 2D NumPy/OpenCV grayscale images with the CUDA backend.

    This wrapper hides the Fortran-order reshape expected by the native
    CUDA registration function.  For defect detection, the default direction
    warps the template/fixed image into the moving image space so defects in
    the inspected image are not resampled away before differencing.

    Arguments
    ---------
    fixed_image : ndarray
        Template/fixed grayscale image in ordinary ``(height, width)`` layout.
    moving_image : ndarray
        Current/moving grayscale image in ordinary ``(height, width)`` layout.
    direction : string or int
        ``"fixed_to_moving"`` / ``"template_to_moving"`` maps the template to
        the inspected image space. ``"moving_to_fixed"`` maps the inspected
        image to the template space. Integer values 1 and 0 are accepted for
        compatibility with the native function.
    preset : string
        Parameter preset. Use ``"defect"`` for conservative defect detection,
        or ``"fast"`` for faster online inspection.
    affine_iterations : sequence of 4 ints
        Multi-resolution affine iteration schedule.
    syn_iterations : sequence of 3 ints
        Multi-resolution SyN iteration schedule.
    robust_threshold : float
        Compatibility parameter retained for existing callers. The iterative
        CUDA LNCC metric performs its own local intensity normalization.
    syn_max_step : float
        Maximum SyN update step.
    syn_max_displacement : float
        Maximum total SyN displacement.
    metric_preprocess : string or None
        Optional internal preprocessing used only for estimating the transform.
        ``"background_subtract"`` removes slow background drift, and
        ``"local_normalize"`` reduces local brightness/contrast mismatch before
        metric computation. ``"auto"`` only removes strong low-frequency
        illumination drift; local contrast is handled by the iterative CUDA
        LNCC metric. The returned image is still the warped original template.
    metric_sigma : float
        Radius-like scale for ``"background_subtract"``.
    fixed_metric_image : ndarray or None
        Explicit fixed metric image. Overrides ``metric_preprocess`` for fixed.
    moving_metric_image : ndarray or None
        Explicit moving metric image. Overrides ``metric_preprocess`` for moving.
    output_dtype : dtype or None
        Output dtype. Use ``None`` to keep the native integer output.
    profile : bool
        Print native CUDA/ITK stage timings. This is intended for performance
        diagnosis and does not change the registration path.
    return_transform : bool
        Also return the final composite sampling map. The map has shape
        ``(height, width, 2)`` and follows the OpenCV convention: component 0
        is the source x/column coordinate and component 1 is the source y/row
        coordinate. Invalid output pixels contain ``(-1, -1)``.

    Returns
    -------
    ndarray or tuple
        Registered image in ordinary ``(height, width)`` layout. When
        ``return_transform=True``, returns ``(registered, transform_map)``.
    """

    fixed = _as_registration_array(fixed_image, "fixed_image")
    moving = _as_registration_array(moving_image, "moving_image")

    if fixed.shape != moving.shape:
        raise ValueError("fixed_image and moving_image must have the same shape")

    params = _resolve_preset(
        preset,
        affine_iterations,
        syn_iterations,
        robust_threshold,
        syn_max_step,
        syn_max_displacement,
    )
    affine_iterations = params["affine_iterations"]
    syn_iterations = params["syn_iterations"]
    robust_threshold = params["robust_threshold"]
    syn_max_step = params["syn_max_step"]
    syn_max_displacement = params["syn_max_displacement"]

    if len(affine_iterations) != 4:
        raise ValueError("affine_iterations must contain 4 values")
    if len(syn_iterations) != 3:
        raise ValueError("syn_iterations must contain 3 values")

    height, width = fixed.shape
    output_direction = _normalize_direction(direction)

    # The native CUDA wrapper follows the historical ANTsPy-GPU layout:
    # Python supplies (width, height) in Fortran order and reshapes the flat
    # native result back to (height, width) in the same order.
    fixed_native = np.asarray(fixed, dtype=np.float32).reshape((width, height), order="F")
    moving_native = np.asarray(moving, dtype=np.float32).reshape((width, height), order="F")

    if metric_preprocess is not None and str(metric_preprocess).lower() == "auto":
        metric_preprocess = _choose_metric_preprocess(fixed, moving)

    if fixed_metric_image is None:
        fixed_metric = _make_metric_image(fixed, metric_preprocess, metric_sigma, "fixed")
    else:
        fixed_metric = _as_registration_array(fixed_metric_image, "fixed_metric_image")
    if moving_metric_image is None:
        moving_metric = _make_metric_image(moving, metric_preprocess, metric_sigma, "moving")
    else:
        moving_metric = _as_registration_array(moving_metric_image, "moving_metric_image")

    if fixed_metric is not None and fixed_metric.shape != fixed.shape:
        raise ValueError("fixed_metric_image must have the same shape as fixed_image")
    if moving_metric is not None and moving_metric.shape != moving.shape:
        raise ValueError("moving_metric_image must have the same shape as moving_image")

    fixed_metric_native = None
    moving_metric_native = None
    if fixed_metric is not None:
        fixed_metric_native = np.asarray(fixed_metric, dtype=np.float32).reshape((width, height), order="F")
    if moving_metric is not None:
        moving_metric_native = np.asarray(moving_metric, dtype=np.float32).reshape((width, height), order="F")

    cuda_registration = get_lib_fn("cudaRegistration")
    result = cuda_registration(
        fixed_native,
        moving_native,
        output_direction,
        [int(v) for v in affine_iterations],
        [int(v) for v in syn_iterations],
        float(robust_threshold),
        float(syn_max_step),
        float(syn_max_displacement),
        fixed_metric_native,
        moving_metric_native,
        bool(profile),
        bool(return_transform),
    )

    if return_transform:
        result, native_map = result

    registered = np.asarray(result).reshape((height, width), order="F")
    if output_dtype is not None:
        registered = registered.astype(output_dtype, copy=False)

    if not return_transform:
        return registered

    # Native CUDA coordinates are transposed by the historical Fortran layout:
    # native x is a Python row, and native y is a Python column. Convert them to
    # an OpenCV-ready absolute sampling map without changing the registration.
    native_map = np.asarray(native_map, dtype=np.float32).reshape((-1, 2))
    source_rows = native_map[:, 0].reshape((height, width), order="F")
    source_columns = native_map[:, 1].reshape((height, width), order="F")
    transform_map = np.empty((height, width, 2), dtype=np.float32)
    transform_map[..., 0] = source_columns
    transform_map[..., 1] = source_rows
    return registered, transform_map
