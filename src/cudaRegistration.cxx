#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "cudaRegistration.cuh"

#include "itkCenteredTransformInitializer.h"
#include "itkEuler2DTransform.h"
#include "itkHistogramMatchingImageFilter.h"
#include "itkImageToHistogramFilter.h"
#include "itkImportImageContainer.h"
#include "itkIntensityWindowingImageFilter.h"
#include "itkSize.h"

namespace nb = nanobind;
using namespace nb::literals;

template <typename ImageType>
typename ImageType::Pointer
PreprocessImage(typename ImageType::ConstPointer inputImage,
                typename ImageType::PixelType lowerScaleValue,
                typename ImageType::PixelType upperScaleValue,
                float winsorizeLowerQuantile,
                float winsorizeUpperQuantile,
                typename ImageType::ConstPointer histogramMatchSourceImage = nullptr)
{
    typedef itk::Statistics::ImageToHistogramFilter<ImageType> HistogramFilterType;
    typedef typename HistogramFilterType::InputBooleanObjectType InputBooleanObjectType;
    typedef typename HistogramFilterType::HistogramSizeType HistogramSizeType;

    HistogramSizeType histogramSize(1);
    histogramSize[0] = 256;

    typename InputBooleanObjectType::Pointer autoMinMaxInputObject = InputBooleanObjectType::New();
    autoMinMaxInputObject->Set(true);

    typename HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();
    histogramFilter->SetInput(inputImage);
    histogramFilter->SetAutoMinimumMaximumInput(autoMinMaxInputObject);
    histogramFilter->SetHistogramSize(histogramSize);
    histogramFilter->SetMarginalScale(10.0);
    histogramFilter->Update();

    float lowerValue = histogramFilter->GetOutput()->Quantile(0, winsorizeLowerQuantile);
    float upperValue = histogramFilter->GetOutput()->Quantile(0, winsorizeUpperQuantile);

    typedef itk::IntensityWindowingImageFilter<ImageType, ImageType> IntensityWindowingImageFilterType;

    typename IntensityWindowingImageFilterType::Pointer windowingFilter = IntensityWindowingImageFilterType::New();
    windowingFilter->SetInput(inputImage);
    windowingFilter->SetWindowMinimum(lowerValue);
    windowingFilter->SetWindowMaximum(upperValue);
    windowingFilter->SetOutputMinimum(lowerScaleValue);
    windowingFilter->SetOutputMaximum(upperScaleValue);
    windowingFilter->Update();

    typename ImageType::Pointer outputImage = nullptr;
    if (histogramMatchSourceImage)
    {
        typedef itk::HistogramMatchingImageFilter<ImageType, ImageType> HistogramMatchingFilterType;
        typename HistogramMatchingFilterType::Pointer matchingFilter = HistogramMatchingFilterType::New();
        matchingFilter->SetSourceImage(windowingFilter->GetOutput());
        matchingFilter->SetReferenceImage(histogramMatchSourceImage);
        matchingFilter->SetNumberOfHistogramLevels(256);
        matchingFilter->SetNumberOfMatchPoints(12);
        matchingFilter->ThresholdAtMeanIntensityOn();
        matchingFilter->Update();

        outputImage = matchingFilter->GetOutput();
    }
    else
    {
        outputImage = windowingFilter->GetOutput();
    }

    outputImage->Update();
    outputImage->DisconnectPipeline();
    return outputImage;
}

using FloatArray2D = nb::ndarray<const float, nb::numpy, nb::ndim<2>, nb::any_contig, nb::device::cpu>;
nb::object cudaRegistration(nb::object fixedImageInput,
                            nb::object movingImageInput,
                            int outputDirection = 0,
                            std::vector<int> affineIterations = {2100, 1200, 1200, 0},
                            std::vector<int> synIterations = {40, 20, 0},
                            double robustThreshold = 0.25,
                            double synMaxStep = 0.2,
                            double synMaxDisplacement = 6.0,
                            nb::object fixedMetricImageInput = nb::none(),
                            nb::object movingMetricImageInput = nb::none(),
                            bool profile = false,
                            bool returnTransform = false)
{
    typedef std::chrono::high_resolution_clock ClockType;
    auto profileStart = ClockType::now();
    auto profileLast = profileStart;
    auto printProfileStage = [&](const char *stage) {
        if (!profile)
        {
            return;
        }
        auto now = ClockType::now();
        std::chrono::duration<double, std::milli> elapsed = now - profileLast;
        std::cout << "[cudaRegistration.profile] wrapper." << stage
                  << "_ms=" << elapsed.count() << std::endl;
        profileLast = now;
    };

    if (outputDirection != 0 && outputDirection != 1)
    {
        throw std::runtime_error("outputDirection must be 0 (moving->fixed) or 1 (fixed->moving)");
    }
    if (affineIterations.size() != 4)
    {
        throw std::runtime_error("affineIterations must contain 4 values");
    }
    if (synIterations.size() != 3)
    {
        throw std::runtime_error("synIterations must contain 3 values");
    }
    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object fixedObject = numpy.attr("asarray")(fixedImageInput, numpy.attr("float32"), "K");
    nb::object movingObject = numpy.attr("asarray")(movingImageInput, numpy.attr("float32"), "K");
    nb::object fixedMetricObject = fixedObject;
    nb::object movingMetricObject = movingObject;
    if (!fixedMetricImageInput.is_none())
    {
        fixedMetricObject = numpy.attr("asarray")(fixedMetricImageInput, numpy.attr("float32"), "K");
    }
    if (!movingMetricImageInput.is_none())
    {
        movingMetricObject = numpy.attr("asarray")(movingMetricImageInput, numpy.attr("float32"), "K");
    }
    FloatArray2D fixedImageArray = nb::cast<FloatArray2D>(fixedObject);
    FloatArray2D movingImageArray = nb::cast<FloatArray2D>(movingObject);
    FloatArray2D fixedMetricImageArray = nb::cast<FloatArray2D>(fixedMetricObject);
    FloatArray2D movingMetricImageArray = nb::cast<FloatArray2D>(movingMetricObject);
    printProfileStage("numpy_cast");

    if (fixedImageArray.shape(0) != movingImageArray.shape(0) ||
        fixedImageArray.shape(1) != movingImageArray.shape(1))
    {
        throw std::runtime_error("fixedImageArray and movingImageArray must have the same shape");
    }
    if (fixedMetricImageArray.shape(0) != fixedImageArray.shape(0) ||
        fixedMetricImageArray.shape(1) != fixedImageArray.shape(1) ||
        movingMetricImageArray.shape(0) != movingImageArray.shape(0) ||
        movingMetricImageArray.shape(1) != movingImageArray.shape(1))
    {
        throw std::runtime_error("metric images must have the same shape as fixedImageArray and movingImageArray");
    }

    int w = static_cast<int>(fixedImageArray.shape(0));
    int h = static_cast<int>(fixedImageArray.shape(1));

    typedef float PixelType;
    const unsigned int Dimension = 2;
    typedef itk::Image<PixelType, Dimension> FixedImageType;
    typedef itk::Image<PixelType, Dimension> MovingImageType;
    typedef itk::Size<Dimension> SizeType;
    typedef itk::ImportImageContainer<SizeType::SizeValueType, PixelType> PixelContainer;

    SizeType size;
    size[0] = h;
    size[1] = w;

    FixedImageType::Pointer fixedImage = FixedImageType::New();
    PixelContainer::Pointer fixedContainer = PixelContainer::New();
    fixedContainer->SetImportPointer(const_cast<float *>(fixedImageArray.data()), w * h);
    fixedImage->SetRegions(size);
    fixedImage->SetPixelContainer(fixedContainer.GetPointer());

    MovingImageType::Pointer movingImage = MovingImageType::New();
    PixelContainer::Pointer movingContainer = PixelContainer::New();
    movingContainer->SetImportPointer(const_cast<float *>(movingImageArray.data()), w * h);
    movingImage->SetRegions(size);
    movingImage->SetPixelContainer(movingContainer.GetPointer());

    FixedImageType::Pointer fixedMetricImage = FixedImageType::New();
    PixelContainer::Pointer fixedMetricContainer = PixelContainer::New();
    fixedMetricContainer->SetImportPointer(const_cast<float *>(fixedMetricImageArray.data()), w * h);
    fixedMetricImage->SetRegions(size);
    fixedMetricImage->SetPixelContainer(fixedMetricContainer.GetPointer());

    MovingImageType::Pointer movingMetricImage = MovingImageType::New();
    PixelContainer::Pointer movingMetricContainer = PixelContainer::New();
    movingMetricContainer->SetImportPointer(const_cast<float *>(movingMetricImageArray.data()), w * h);
    movingMetricImage->SetRegions(size);
    movingMetricImage->SetPixelContainer(movingMetricContainer.GetPointer());
    printProfileStage("itk_import");

    // Estimate the transform from metric images, but keep original images for final resampling.
    FixedImageType::Pointer preprocessFixedImage = PreprocessImage<FixedImageType>(
        fixedMetricImage.GetPointer(), 0, 1, 0, 1, nullptr);

    MovingImageType::Pointer preprocessMovingImage = PreprocessImage<MovingImageType>(
        movingMetricImage.GetPointer(), 0, 1, 0, 1, preprocessFixedImage.GetPointer());
    printProfileStage("itk_preprocess");

    typedef itk::Euler2DTransform<float> TransformType;
    typename TransformType::Pointer transform = TransformType::New();

    typedef itk::CenteredTransformInitializer<TransformType, FixedImageType, MovingImageType> TransformInitializerType;
    typename TransformInitializerType::Pointer initializer = TransformInitializerType::New();
    initializer->SetTransform(transform);
    initializer->SetFixedImage(fixedImage);
    initializer->SetMovingImage(movingImage);
    initializer->MomentsOn();
    initializer->InitializeTransform();
    printProfileStage("initial_transform");

    std::unique_ptr<float[]> transformMap;
    if (returnTransform)
    {
        transformMap = std::make_unique<float[]>(w * h * 2);
    }

    std::unique_ptr<int[]> result(itk::cudaRegistrationStart<FixedImageType, MovingImageType, TransformType>(
        preprocessFixedImage, preprocessMovingImage, transform, fixedImage, movingImage, outputDirection,
        affineIterations[0], affineIterations[1], affineIterations[2], affineIterations[3],
        synIterations[0], synIterations[1], synIterations[2], robustThreshold, synMaxStep,
        synMaxDisplacement, profile, transformMap.get()));
    printProfileStage("native_total");

    if (!result)
    {
        throw std::runtime_error("cudaRegistration returned a null result");
    }

    // Match the old pybind wrapper by returning a flat NumPy array that Python reshapes.
    nb::object outputObject = numpy.attr("empty")(w * h, numpy.attr("int32"));
    auto outputArray = nb::cast<nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>>(outputObject);
    std::copy(result.get(), result.get() + w * h, outputArray.data());
    printProfileStage("output_copy");
    if (profile)
    {
        auto now = ClockType::now();
        std::chrono::duration<double, std::milli> elapsed = now - profileStart;
        std::cout << "[cudaRegistration.profile] wrapper.total_ms="
                  << elapsed.count() << std::endl;
    }
    if (!returnTransform)
    {
        return outputObject;
    }

    // Return the native interleaved coordinate map. Python converts its axis
    // order to the conventional OpenCV (source x, source y) map.
    nb::object mapObject = numpy.attr("empty")(w * h * 2, numpy.attr("float32"));
    auto mapArray = nb::cast<nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::device::cpu>>(mapObject);
    std::copy(transformMap.get(), transformMap.get() + w * h * 2, mapArray.data());
    return nb::make_tuple(outputObject, mapObject);
}

void wrap_cudaRegistration(nb::module_ &m)
{
    m.def("cudaRegistration", &cudaRegistration,
          "fixedImageArray"_a,
          "movingImageArray"_a,
          "outputDirection"_a = 0,
          "affineIterations"_a = std::vector<int>{2100, 1200, 1200, 0},
          "synIterations"_a = std::vector<int>{40, 20, 0},
          "robustThreshold"_a = 0.25,
          "synMaxStep"_a = 0.2,
          "synMaxDisplacement"_a = 6.0,
          "fixedMetricImageArray"_a = nb::none(),
          "movingMetricImageArray"_a = nb::none(),
          "profile"_a = false,
          "returnTransform"_a = false);
}
