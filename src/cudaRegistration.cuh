#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <iostream>
#include "itkSmoothingRecursiveGaussianImageFilter.h"

#ifndef itkCudaRegistration_h
#define itkCudaRegistration_h

namespace itk{

// SyN displacement/update fields are bandwidth-bound and only need pixel-level precision.
// Keep affine parameters and reductions in double, but store dense fields as float.
using CudaFieldValueType = float;

// Affine metric scratch buffers dominate memory traffic during line search.
// Store per-point derivatives/values in float, while keeping transform state
// and final reductions in double for stable optimizer updates.
using CudaAffineMetricValueType = float;

void cudaStartAffinePerLevel(int w, int h, int spacing, float *d_FixedImage, float *d_MovingImage, double *d_InitialTransform,
                             double *d_AffineTransform, double offset_x, double offset_y, int maxIteration, int minIteration,
                             bool profile = false, const char *profileLabel = "");

void EXEC_cudaInitDisplacementField(int w, int h, void *d_DisplacementField);

void cudaStartSynPerLevel(int w, int h, int spacing, float *d_FixedImage, float *d_MovingImage, double *d_InitialTransform,
                          double *d_AffineTransform, CudaFieldValueType *d_FixedField, CudaFieldValueType *d_FixedInverseField, CudaFieldValueType *d_MovingField,
                          CudaFieldValueType *d_MovingInverseField, double offset_x, double offset_y, int maxIteration,
                          int minIteration, double robustThreshold, double synMaxStep, double synMaxDisplacement,
                          bool profile = false, const char *profileLabel = "");

void EXEC_cudaAdaptDisplacementField(int w, int h, int spacingOld, int spacingNew, void *d_OldField, double offsetOld_x,
                                     double offsetOld_y, double offsetNew_x, double offsetNew_y, void *d_NewField);

void EXEC_cudaComposeField(int w, int h, int spacing, void *d_DisplacementField, void *d_WarpingField, double offset_x,
                           double offset_y, void *t_OutputField);

void EXEC_cudaApplyTransform(int w, int h, void *d_OrigImage, void *d_InitialTransform, void *d_AffineTransform,
                             void *d_OutputFieldFinal, void *d_OutputInverseFieldFinal, int outputDirection,
                             void *d_ResultImage, void *d_TransformMap = nullptr);

template <typename ImageType>
typename ImageType::Pointer SmoothImage(typename ImageType::Pointer inputImage, double sigma){

    using ImageSmoothingFilterType = SmoothingRecursiveGaussianImageFilter<ImageType, ImageType>;
    typename ImageSmoothingFilterType::Pointer imageSmoothingFilter = ImageSmoothingFilterType::New();

    double imageSigmaArray[2];
    imageSigmaArray[0] = sigma;
    imageSigmaArray[1] = sigma;

    imageSmoothingFilter->SetSigmaArray(imageSigmaArray); // [3,3]-[2,2]-[1,1]
    imageSmoothingFilter->SetInput(inputImage);
    imageSmoothingFilter->Update();
    // imageSmoothingFilter->GetOutput()->DisconnectPipeline();

    typename ImageType::Pointer outputImage = nullptr;

    outputImage = imageSmoothingFilter->GetOutput();
    imageSmoothingFilter->Update();
    imageSmoothingFilter->GetOutput()->DisconnectPipeline();

    return outputImage;
}

template <class FixedImageType, class MovingImageType, class InitialTransformType>
int* cudaRegistrationStart(FixedImageType *m_FixedImage, MovingImageType *m_MovingImage, InitialTransformType *initialTransform,
                           FixedImageType *m_FixedOrigImage, MovingImageType *m_MovingOrigImage, int outputDirection = 0,
                           int affineIteration0 = 2100, int affineIteration1 = 1200, int affineIteration2 = 1200,
                           int affineIteration3 = 0, int synIteration0 = 40, int synIteration1 = 20,
                           int synIteration2 = 0, double robustThreshold = 0.25, double synMaxStep = 0.2,
                           double synMaxDisplacement = 6.0, bool profile = false,
                           float *m_TransformMap = nullptr){

    typedef std::chrono::high_resolution_clock ClockType;
    auto profileStart = ClockType::now();
    auto profileLast = profileStart;
    auto printProfileStage = [&](const char *stage) {
        if (!profile)
        {
            return;
        }
        // CUDA work is asynchronous; synchronize only when profiling is enabled.
        cudaDeviceSynchronize();
        auto now = ClockType::now();
        std::chrono::duration<double, std::milli> elapsed = now - profileLast;
        std::cout << "[cudaRegistration.profile] native." << stage
                  << "_ms=" << elapsed.count() << std::endl;
        profileLast = now;
    };

    typedef typename FixedImageType::PixelType FixedPixelType;
    typedef typename MovingImageType::PixelType MovingPixelType;

    int w, h;
    typename FixedImageType::SizeType FixedImageSize =
        m_FixedImage->GetLargestPossibleRegion().GetSize();
    w = FixedImageSize[0];  // 198 or 416
    h = FixedImageSize[1];  // 640 or 1344

    double offset_x = 2.5;
    if(w == 416){offset_x = 1.5;}

    // std::cout << "w:" << w << " " << "h:" << h << std::endl;

    /*-------------------------------------------------------------------------------------------------------------*/

    FixedPixelType *d_FixedImage;
    cudaMalloc((void**) &d_FixedImage, sizeof(FixedPixelType) * w * h);

    /*-------------------------------------------------------------------------------------------------------------*/

    MovingPixelType *d_MovingImage;
    cudaMalloc((void**) &d_MovingImage, sizeof(MovingPixelType) * w * h);

    /*-------------------------------------------------------------------------------------------------------------*/

    double temp_MovingInitialTransform[8];  // 前4个值是Matrix,后2个值是Offset,最后两个值是Center
    temp_MovingInitialTransform[0] = initialTransform->GetMatrix()[0][0];
    temp_MovingInitialTransform[1] = initialTransform->GetMatrix()[0][1];
    temp_MovingInitialTransform[2] = initialTransform->GetMatrix()[1][0];
    temp_MovingInitialTransform[3] = initialTransform->GetMatrix()[1][1];
    temp_MovingInitialTransform[4] = initialTransform->GetOffset()[0];
    temp_MovingInitialTransform[5] = initialTransform->GetOffset()[1];
    temp_MovingInitialTransform[6] = initialTransform->GetCenter()[0];
    temp_MovingInitialTransform[7] = initialTransform->GetCenter()[1];

//     std::cout << "initialTransform:[" << initialTransform->GetMatrix()[0][0] << "," << initialTransform->GetMatrix()[0][1]
//                                << "," << initialTransform->GetMatrix()[1][0] << "," << initialTransform->GetMatrix()[1][1]
//                               << "][" << initialTransform->GetOffset()[0] << "," << initialTransform->GetOffset()[1]
//                               << "][" << initialTransform->GetCenter()[0] << "," << initialTransform->GetCenter()[1]
//                                << "]" << std::endl;

    double *d_InitialTransform;
    cudaMalloc((void**) &d_InitialTransform, sizeof(double) * 8);
    cudaMemcpy(d_InitialTransform, temp_MovingInitialTransform, sizeof(double) * 8, cudaMemcpyHostToDevice);

    /*-------------------------------------------------------------------------------------------------------------*/

    double temp_MovingAffineTransform[8];  // 前4个值是Matrix,后2个值是Offset,最后两个值是Center
    temp_MovingAffineTransform[0] = 1; temp_MovingAffineTransform[1] = 0;
    temp_MovingAffineTransform[2] = 0; temp_MovingAffineTransform[3] = 1;
    temp_MovingAffineTransform[4] = 0; temp_MovingAffineTransform[5] = 0;
    temp_MovingAffineTransform[6] = initialTransform->GetCenter()[0];
    temp_MovingAffineTransform[7] = initialTransform->GetCenter()[1];

    double *d_AffineTransform;
    cudaMalloc((void**) &d_AffineTransform, sizeof(double) * 8);
    cudaMemcpy(d_AffineTransform, temp_MovingAffineTransform, sizeof(double) * 8, cudaMemcpyHostToDevice);
    printProfileStage("setup_device");

    /*-------------------------------------------------------------------------------------------------------------*/

    typedef typename FixedImageType::Pointer FixedImagePointer;
    FixedImagePointer t_FixedImage;
    typedef typename MovingImageType::Pointer MovingImagePointer;
    MovingImagePointer t_MovingImage;

    /*-------------------------------------------------------------------------------------------------------------*/

    t_FixedImage = SmoothImage<FixedImageType>(m_FixedImage, 3);
    t_MovingImage = SmoothImage<MovingImageType>(m_MovingImage, 3);

    cudaMemcpy(d_FixedImage, t_FixedImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaMemcpy(d_MovingImage, t_MovingImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);

    if (affineIteration0 > 0)
    {
        cudaStartAffinePerLevel(w, h, 4, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
                                offset_x, 1.5, affineIteration0, 10, profile, "affine_level0");
    }
    printProfileStage("affine_level0");

//     std::cout << "---------------------------------------------------------------------------------" << std::endl;

    /*-------------------------------------------------------------------------------------------------------------*/

    t_FixedImage = SmoothImage<FixedImageType>(m_FixedImage, 2);
    t_MovingImage = SmoothImage<MovingImageType>(m_MovingImage, 2);

    cudaMemcpy(d_FixedImage, t_FixedImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaMemcpy(d_MovingImage, t_MovingImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);

    if (affineIteration1 > 0)
    {
        cudaStartAffinePerLevel(w, h, 2, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
                                0.5, 0.5, affineIteration1, 10, profile, "affine_level1");
    }
    printProfileStage("affine_level1");

//     std::cout << "---------------------------------------------------------------------------------" << std::endl;

    /*-------------------------------------------------------------------------------------------------------------*/

    t_FixedImage = SmoothImage<FixedImageType>(m_FixedImage, 1);
    t_MovingImage = SmoothImage<MovingImageType>(m_MovingImage, 1);

    cudaMemcpy(d_FixedImage, t_FixedImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaMemcpy(d_MovingImage, t_MovingImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);

    if (affineIteration2 > 0)
    {
        cudaStartAffinePerLevel(w, h, 2, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
                                0.5, 0.5, affineIteration2, 10, profile, "affine_level2");
    }
    printProfileStage("affine_level2");

//     std::cout << "---------------------------------------------------------------------------------" << std::endl;

    /*-------------------------------------------------------------------------------------------------------------*/

    cudaMemcpy(d_FixedImage, m_FixedImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaMemcpy(d_MovingImage, m_MovingImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);

    // CPU SyN defaults to a 0-iteration full-resolution affine level; skip it when disabled.
    if (affineIteration3 > 0)
    {
        cudaStartAffinePerLevel(w, h, 1, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
                                0.0, 0.0, affineIteration3, 10, profile, "affine_level3");
    }
    printProfileStage("affine_level3");

    /*-------------------------------------------------------------------------------------------------------------*/

    int spacing;

    spacing = 4;

    CudaFieldValueType *d_FixedField;
    cudaMalloc((void**) &d_FixedField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaInitDisplacementField(w / spacing, h / spacing, d_FixedField);

    CudaFieldValueType *d_FixedInverseField;
    cudaMalloc((void**) &d_FixedInverseField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaInitDisplacementField(w / spacing, h / spacing, d_FixedInverseField);

    CudaFieldValueType *d_MovingField;
    cudaMalloc((void**) &d_MovingField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaInitDisplacementField(w / spacing, h / spacing, d_MovingField);

    CudaFieldValueType *d_MovingInverseField;
    cudaMalloc((void**) &d_MovingInverseField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaInitDisplacementField(w / spacing, h / spacing, d_MovingInverseField);

    t_FixedImage = SmoothImage<FixedImageType>(m_FixedImage, 2);
    t_MovingImage = SmoothImage<MovingImageType>(m_MovingImage, 2);

    cudaMemcpy(d_FixedImage, t_FixedImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaMemcpy(d_MovingImage, t_MovingImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaStartSynPerLevel(w, h, spacing, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
        d_FixedField, d_FixedInverseField, d_MovingField, d_MovingInverseField, offset_x, 1.5, synIteration0, 8,
        robustThreshold, synMaxStep, synMaxDisplacement, profile, "syn_level0");
    printProfileStage("syn_level0");

    // std::cout << "---------------------------------------------------------------------------------" << std::endl;

    // /*-------------------------------------------------------------------------------------------------------------*/

    spacing = 2;

    CudaFieldValueType *d_FixedFieldNew;
    cudaMalloc((void**) &d_FixedFieldNew, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaAdaptDisplacementField(w, h, 4, spacing, d_FixedField, offset_x, 1.5, 0.5, 0.5, d_FixedFieldNew);

    CudaFieldValueType *d_FixedInverseFieldNew;
    cudaMalloc((void**) &d_FixedInverseFieldNew, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaAdaptDisplacementField(w, h, 4, spacing, d_FixedInverseField, offset_x, 1.5, 0.5, 0.5, d_FixedInverseFieldNew);

    CudaFieldValueType *d_MovingFieldNew;
    cudaMalloc((void**) &d_MovingFieldNew, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaAdaptDisplacementField(w, h, 4, spacing, d_MovingField, offset_x, 1.5, 0.5, 0.5, d_MovingFieldNew);

    CudaFieldValueType *d_MovingInverseFieldNew;
    cudaMalloc((void**) &d_MovingInverseFieldNew, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
    EXEC_cudaAdaptDisplacementField(w, h, 4, spacing, d_MovingInverseField, offset_x, 1.5, 0.5, 0.5, d_MovingInverseFieldNew);

    cudaFree(d_FixedField);cudaFree(d_FixedInverseField);cudaFree(d_MovingField);cudaFree(d_MovingInverseField);
    printProfileStage("adapt_4_to_2");

    t_FixedImage = SmoothImage<FixedImageType>(m_FixedImage, 1);
    t_MovingImage = SmoothImage<MovingImageType>(m_MovingImage, 1);

    cudaMemcpy(d_FixedImage, t_FixedImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaMemcpy(d_MovingImage, t_MovingImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);

    cudaStartSynPerLevel(w, h, spacing, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
        d_FixedFieldNew, d_FixedInverseFieldNew, d_MovingFieldNew, d_MovingInverseFieldNew, 0.5, 0.5, synIteration1, 8,
        robustThreshold, synMaxStep, synMaxDisplacement, profile, "syn_level1");
    printProfileStage("syn_level1");

    // std::cout << "---------------------------------------------------------------------------------" << std::endl;

    // /*-------------------------------------------------------------------------------------------------------------*/

    spacing = 1;

    CudaFieldValueType *d_FixedFieldFinal = nullptr;
    CudaFieldValueType *d_FixedInverseFieldFinal = nullptr;
    CudaFieldValueType *d_MovingFieldFinal = nullptr;
    CudaFieldValueType *d_MovingInverseFieldFinal = nullptr;

    if (synIteration2 > 0)
    {
        // Full-resolution SyN still needs the complete symmetric field set.
        cudaMalloc((void**) &d_FixedFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_FixedFieldNew, 0.5, 0.5, 0, 0, d_FixedFieldFinal);

        cudaMalloc((void**) &d_FixedInverseFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_FixedInverseFieldNew, 0.5, 0.5, 0, 0, d_FixedInverseFieldFinal);

        cudaMalloc((void**) &d_MovingFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_MovingFieldNew, 0.5, 0.5, 0, 0, d_MovingFieldFinal);

        cudaMalloc((void**) &d_MovingInverseFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_MovingInverseFieldNew, 0.5, 0.5, 0, 0, d_MovingInverseFieldFinal);
    }
    else if (outputDirection == 1)
    {
        // Defect detection normally warps the fixed/template image to moving space.
        // With no full-resolution SyN iterations, only these two fields are needed
        // to compose the final inverse output field.
        cudaMalloc((void**) &d_FixedInverseFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_FixedInverseFieldNew, 0.5, 0.5, 0, 0, d_FixedInverseFieldFinal);

        cudaMalloc((void**) &d_MovingFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_MovingFieldNew, 0.5, 0.5, 0, 0, d_MovingFieldFinal);
    }
    else
    {
        // Legacy moving-to-fixed output uses the complementary field pair.
        cudaMalloc((void**) &d_FixedFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_FixedFieldNew, 0.5, 0.5, 0, 0, d_FixedFieldFinal);

        cudaMalloc((void**) &d_MovingInverseFieldFinal, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);
        EXEC_cudaAdaptDisplacementField(w, h, 2, spacing, d_MovingInverseFieldNew, 0.5, 0.5, 0, 0, d_MovingInverseFieldFinal);
    }

    cudaFree(d_FixedFieldNew);cudaFree(d_FixedInverseFieldNew);cudaFree(d_MovingFieldNew);cudaFree(d_MovingInverseFieldNew);
    printProfileStage("adapt_2_to_1");

    if (synIteration2 > 0)
    {
        cudaMemcpy(d_FixedImage, m_FixedImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);
        cudaMemcpy(d_MovingImage, m_MovingImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);
        cudaStartSynPerLevel(w, h, spacing, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
            d_FixedFieldFinal, d_FixedInverseFieldFinal, d_MovingFieldFinal, d_MovingInverseFieldFinal, 0, 0,
            synIteration2, 8, robustThreshold, synMaxStep, synMaxDisplacement, profile, "syn_level2");
    }
    // A zero full-resolution SyN stage is intentionally skipped to avoid buffer churn.
    printProfileStage("syn_level2");

    // /*-------------------------------------------------------------------------------------------------------------*/

    CudaFieldValueType *d_OutputFieldFinal = nullptr;
    CudaFieldValueType *d_OutputInverseFieldFinal = nullptr;
    if (outputDirection == 1)
    {
        cudaMalloc((void**) &d_OutputInverseFieldFinal, sizeof(CudaFieldValueType) * w * h * 2);
        EXEC_cudaComposeField(w, h, spacing, d_FixedInverseFieldFinal, d_MovingFieldFinal, 0, 0, d_OutputInverseFieldFinal);
    }
    else
    {
        cudaMalloc((void**) &d_OutputFieldFinal, sizeof(CudaFieldValueType) * w * h * 2);
        EXEC_cudaComposeField(w, h, spacing, d_MovingInverseFieldFinal, d_FixedFieldFinal, 0, 0, d_OutputFieldFinal);
    }
    printProfileStage("output_compose");

    if (d_FixedFieldFinal) { cudaFree(d_FixedFieldFinal); }
    if (d_FixedInverseFieldFinal) { cudaFree(d_FixedInverseFieldFinal); }
    if (d_MovingFieldFinal) { cudaFree(d_MovingFieldFinal); }
    if (d_MovingInverseFieldFinal) { cudaFree(d_MovingInverseFieldFinal); }

    // /*-------------------------------------------------------------------------------------------------------------*/

    void *d_OrigImage = nullptr;
    if (outputDirection == 1)
    {
        cudaMalloc((void**) &d_OrigImage, sizeof(FixedPixelType) * w * h);
        cudaMemcpy(d_OrigImage, m_FixedOrigImage->GetBufferPointer(), sizeof(FixedPixelType) * w * h, cudaMemcpyHostToDevice);
    }
    else
    {
        cudaMalloc((void**) &d_OrigImage, sizeof(MovingPixelType) * w * h);
        cudaMemcpy(d_OrigImage, m_MovingOrigImage->GetBufferPointer(), sizeof(MovingPixelType) * w * h, cudaMemcpyHostToDevice);
    }

    int *d_ResultImage;
    cudaMalloc((void**) &d_ResultImage, sizeof(int) * w * h);

    float *d_TransformMap = nullptr;
    if (m_TransformMap)
    {
        cudaMalloc((void**) &d_TransformMap, sizeof(float) * w * h * 2);
    }

    // Defect detection should compare the untouched moving image with a warped template.
    // The legacy warped-moving output remains the default for backward compatibility.
    EXEC_cudaApplyTransform(w, h, d_OrigImage, d_InitialTransform, d_AffineTransform, d_OutputFieldFinal,
                            d_OutputInverseFieldFinal, outputDirection, d_ResultImage, d_TransformMap);

    /*-------------------------------------------------------------------------------------------------------------*/

    int* m_ResultImage = new int[w * h];
    cudaMemcpy(m_ResultImage, d_ResultImage, sizeof(int) * w * h, cudaMemcpyDeviceToHost);
    if (m_TransformMap)
    {
        // The map stores the exact source coordinate used by the final warp.
        // Keeping it interleaved avoids an additional CUDA conversion kernel.
        cudaMemcpy(m_TransformMap, d_TransformMap, sizeof(float) * w * h * 2, cudaMemcpyDeviceToHost);
    }
    printProfileStage("apply_and_download");

    cudaFree(d_FixedImage);
    cudaFree(d_MovingImage);
    cudaFree(d_InitialTransform);
    cudaFree(d_AffineTransform);
    cudaFree(d_OutputFieldFinal);
    cudaFree(d_OutputInverseFieldFinal);
    cudaFree(d_OrigImage);
    cudaFree(d_ResultImage);
    if (d_TransformMap) { cudaFree(d_TransformMap); }

    if (profile)
    {
        auto now = ClockType::now();
        std::chrono::duration<double, std::milli> elapsed = now - profileStart;
        std::cout << "[cudaRegistration.profile] native.total_ms="
                  << elapsed.count() << std::endl;
    }

//     for(int i = 0; i < w * h; i++){
//         m_ResultImage[i] = i % 100;
//     }

    return m_ResultImage;

}

}

#endif
