# ANTsPy GPU Registration Handoff

## 1. Project Context

当前目标是在最新 CPU 版 ANTsPy 代码库中迁移并优化 GPU 配准能力，用于软管印刷缺陷检测场景。

业务路径为：

1. 线阵相机采集图像。
2. 先通过模板匹配定位到短图 ROI。
3. GPU 配准对齐最佳模板图与当前检测图。
4. 配准后做差分检测缺陷。

当前不考虑长图整图配准，也不输出 valid mask。

## 2. Key Algorithm Decisions

### 2.1 输出方向

缺陷检测场景不能把当前检测图作为 moving 图直接 warp 到模板空间后再做差分，否则缺陷区域会被插值和变形，表现为缺陷被“补洞”或失真。

当前推荐路径：

```python
cudaRegistration(fixed_image, moving_image, outputDirection=1, ...)
```

含义：

- `fixed_image`: 模板图
- `moving_image`: 当前检测图
- `outputDirection=1`: 输出配准后的模板图到当前检测图空间

差分应使用：

```python
diff_img = cv2.absdiff(moving_image, ants_img)
```

其中 `ants_img` 是 warped template。

### 2.2 SyN 约束

GPU 版本加入了受限 SyN 思路：

- robust loss 抑制缺陷区域对形变场的影响。
- `synMaxStep` 限制单步更新。
- `synMaxDisplacement` 限制总位移幅度。
- 支持 `synIterations` 参数化，可把最后一层设为 0 来跳过高成本细层。

推荐默认值：

```python
affineIterations = [2100, 1200, 1200, 0]
synIterations = [40, 20, 0]
robustThreshold = 0.25
synMaxStep = 0.2
synMaxDisplacement = 6.0
```

偏速度优先可使用：

```python
affineIterations = [1200, 800, 400, 0]
synIterations = [20, 10, 0]
robustThreshold = 0.25
synMaxStep = 0.15
synMaxDisplacement = 4.0
```

## 3. Current Code Changes

### 3.1 Added GPU Registration Files

新增文件：

- `src/cudaRegistration.cu`
- `src/cudaRegistration.cuh`
- `src/cudaRegistration.cxx`

作用：

- `.cu`: CUDA 计算核心。
- `.cuh`: 模板入口、参数定义、CPU/ITK 与 CUDA 调用桥接。
- `.cxx`: nanobind Python 包装层，暴露 `ants.lib.cudaRegistration`。

### 3.2 Modified Build Integration

已修改：

- `CMakeLists.txt`
- `src/main.cpp`

主要变化：

- CMake 启用 CUDA 语言。
- `src/cudaRegistration.cu` 加入 `lib` 模块构建。
- `main.cpp` include `cudaRegistration.cxx` 并调用 `wrap_cudaRegistration(m)`。
- 默认 CUDA 架构当前设置为 `89`，适合 RTX 40 系列；RTX 30 系列建议用 `86`。

### 3.3 Modified Windows Scripts

已修正：

- `scripts/configure_ITK.bat`
- `scripts/configure_ANTs.bat`

修复点：

- 修复自动转换遗留的 `echo ... REM UNKNOWN` 重定向错误。
- 使用 `rmdir /S /Q` 删除目录，避免 `DEL /S /Q` 留下残缺目录。
- `configure_ITK.bat` 增加 `itksource` 残缺校验。
- `configure_ANTs.bat` 增加 `antscore` 核心文件校验，避免半成品目录导致 CMake 找不到源码。
- 清理临时 `antsrepo`。

## 4. Python Call Path

### 4.1 Friendly Wrapper

已新增 Python 友好接口：

```python
import ants

ants_img = ants.gpu_registration(template_gray, current_gray)
diff_img = cv2.absdiff(current_gray, ants_img)
```

封装位置：

```text
ants/registration/gpu_registration.py
```

接口内部会处理：

- 普通 `(height, width)` 灰度图输入。
- `float32` 转换。
- native CUDA 接口需要的 Fortran-order reshape。
- 默认输出方向 `fixed_to_moving`，即模板图配准到当前检测图空间。
- 默认 affine/SyN/robust 参数。
- 返回普通 `(height, width)` 图像。

速度优先可直接使用 `fast` 预设：

```python
ants_img = ants.gpu_registration(template_gray, current_gray, preset="fast")
```

也可以手动覆盖参数：

```python
ants_img = ants.gpu_registration(
    template_gray,
    current_gray,
    affine_iterations=(1200, 800, 400, 0),
    syn_iterations=(20, 10, 0),
    robust_threshold=0.25,
    syn_max_step=0.15,
    syn_max_displacement=4.0,
)
```

第 4 组数据验证显示，方向受限预设和 full-resolution SyN 对文字/图案没有稳定收益，容易把高频文字、条码、半色调点阵错误吸附。因此当前已撤销 `print_x/print_y/fast_print_x/fast_print_y` 预设，只保留稳定的 `defect` 与 `fast`。

可选 metric 预处理：

```python
ants_img = ants.gpu_registration(
    template_gray,
    current_gray,
    metric_preprocess="local_normalize",
    metric_sigma=15,
)
```

该处理只用于估计 affine + SyN 形变场，最终仍然 warp 原始模板图。因此：

```text
metric 图：用于配准估计
原始模板图：用于最终输出和缺陷差分
```

默认 `metric_preprocess=None`，完全保持当前行为。也可以显式传入自定义 metric 图：

```python
ants_img = ants.gpu_registration(
    template_gray,
    current_gray,
    metric_preprocess="auto",
)
```

`auto` 不会跑多次完整配准，只基于 resize 后模板/检测图的高残差和边缘密度差做低成本判断。当前规则会优先覆盖 test2/test4/test6/test8 这类“残差较高但边缘密度接近”的图像，并避开 test5/test7 这类 normalization 容易变差的数据。

test8 验证中，`local_normalize` 明显优于默认路径：

```text
default: mean=5.1308, p95=17, gt20=3.42%
ln15:    mean=4.3948, p95=13, gt20=1.43%
ln25:    mean=4.3157, p95=12, gt20=1.26%
```

如果图案文字局部残差仍较明显，可手动尝试：

```python
ants_img = ants.gpu_registration(
    template_gray,
    current_gray,
    metric_preprocess="local_normalize",
    metric_sigma=25,
)
```

也可以显式传入自定义 metric 图：

```python
ants_img = ants.gpu_registration(
    template_gray,
    current_gray,
    fixed_metric_image=template_metric,
    moving_metric_image=current_metric,
)
```

### 4.2 Native Entry

新版 ANTsPy 中 `get_lib_fn` 在：

```python
from ants.internal import get_lib_fn
```

旧代码：

```python
from ants import utils
cudaRegistration = utils.get_lib_fn("cudaRegistration")
```

会报：

```text
AttributeError: module 'ants.utils' has no attribute 'get_lib_fn'
```

推荐改为：

```python
from ants.internal import get_lib_fn

cudaRegistration = get_lib_fn("cudaRegistration")
```

或直接：

```python
import ants

cudaRegistration = ants.lib.cudaRegistration
```

如果需要兼容旧测试脚本，可在 `ants/utils/__init__.py` 中重新导出：

```python
from ants.internal import get_lib_fn
```

此兼容改动尚未确认落地。

## 5. Known Build Issues And Fixes

### 5.1 Windows SDK 工具缺失

症状：

```text
CMAKE_MT-NOTFOUND
rc ... failed
```

原因：

MSVC 编译器可用，但 Windows SDK 的 `rc.exe` 和 `mt.exe` 不在环境中。

建议使用：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
```

并检查：

```bat
where cl
where rc
where mt
where nvcc
```

### 5.2 antscore 源文件缺失

症状：

```text
Cannot find source file:
src/antscore/antsUtilities.cxx
src/antscore/antsRegistrationTemplateHeader.cxx
src/antscore/ImageMathHelper2D.cxx
src/antscore/antsAffineInitializer.cxx
```

原因：

旧版 `configure_ANTs.bat` 只判断 `src\antscore` 目录是否存在，没有判断核心文件是否齐全。上次失败留下半成品目录时会跳过重新拷贝。

当前 `configure_ANTs.bat` 已修复。

### 5.3 numpy 重定义

症状：

```text
error C2374: "numpy": redefinition
```

原因：

`src/cudaRegistration.cxx` 中同一函数作用域重复声明：

```cpp
nb::module_ numpy = nb::module_::import_("numpy");
```

当前已删除第二次声明，复用第一次导入的 `numpy`。

## 6. Build Commands

在 VS 2022 x64 Native Tools 环境中执行。

RTX 40 系列：

```bat
cd /d E:\Source\Antspy-GPU\ANTsPy
python -m pip install -U pip build scikit-build-core ninja
set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=89
python -m build --wheel --no-isolation
```

RTX 30 系列：

```bat
cd /d E:\Source\Antspy-GPU\ANTsPy
python -m pip install -U pip build scikit-build-core ninja
set CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=86
python -m build --wheel --no-isolation
```

如遇到残缺源码目录，可清理后重来：

```bat
cd /d E:\Source\Antspy-GPU\ANTsPy
rmdir /s /q build
rmdir /s /q src\antscore
rmdir /s /q src\antsrepo
python -m build --wheel --no-isolation
```

ITK 已构建成功时，不建议删除 `itkbuild`，避免重复耗时编译。

## 7. Test Script Notes

当前测试脚本路径：

```text
E:\yjlg\REG\reg_gpu.py
```

新版调用建议：

```python
from ants.internal import get_lib_fn

cudaRegistration = get_lib_fn("cudaRegistration")
```

配准调用示例：

```python
Gc = cudaRegistration(
    fixed_image,
    moving_image,
    1,
    [1200, 800, 400, 0],
    [20, 10, 0],
    0.25,
    0.15,
    4.0,
)
```

输出 reshape：

```python
ants_img = Gc.reshape((h, w), order="F")
```

差分：

```python
diff_img = cv2.absdiff(moving_image, ants_img)
```

## 8. Remaining Recommendations

1. 增加 `ants.utils.get_lib_fn` 兼容导出，减少旧脚本迁移成本。
2. 增加小尺寸 synthetic 图像 smoke test，验证 `ants.gpu_registration` 可导入并返回正确尺寸。
3. 对软管印刷场景保留两套参数：
   - `quality`: 更稳，适合疑难图。
   - `fast`: 更快，适合在线检测。
4. 编译通过后，用 `E:\yjlg\REG\reg_gpu.py` 对比：
   - 缺陷是否仍被补洞。
   - 边缘黑边是否可接受。
   - 单张 ROI 耗时是否满足节拍。

## 9. Git Status Reminder

当前相关文件预期会显示为修改或新增：

```text
M  CMakeLists.txt
M  scripts/configure_ANTs.bat
M  scripts/configure_ITK.bat
M  src/main.cpp
?? src/cudaRegistration.cu
?? src/cudaRegistration.cuh
?? src/cudaRegistration.cxx
?? GPU_REGISTRATION_HANDOFF.md
```

如果需要让 `itksource` 内部改动显示在 git 中，需要检查 `.gitignore` 规则是否忽略了该目录，并按需增加例外规则。
