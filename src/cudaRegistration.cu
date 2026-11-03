#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudaRegistration.cuh"

#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <stdexcept>

#include <windows.h>

namespace itk{

#define CUDA_CHECK(call)                                                                            \
    do                                                                                              \
    {                                                                                               \
        cudaError_t cudaStatus = (call);                                                            \
        if (cudaStatus != cudaSuccess)                                                              \
        {                                                                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus)                           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;                        \
            throw std::runtime_error(cudaGetErrorString(cudaStatus));                               \
        }                                                                                           \
    } while (0)

inline dim3 CudaBlock2D()
{
    return dim3(16, 16);
}

inline dim3 CudaGrid2D(int width, int height)
{
    dim3 block = CudaBlock2D();
    return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

/*-------------------------------------------------------------------------------------------------------------*/

/*
函数输入参数规范:
函数参数从前往后赢依次为:
1.尺寸参数(w,h,spacing,...)
2.图像参数(d_FixedImage,d_MovingImage,...)
3.变换矩阵(d_AffineTransform,d_InitialTransform,d_DisplacementField,...)
4.其他参数
5.输出结果
6.中间存储(t_Value,...)
*/

/*-------------------------------------------------------------------------------------------------------------*/

double getGaussianRandomValue(){
    double U1 = rand() * 1.0f / RAND_MAX; // 0~1均匀分布
    double U2 = rand() * 1.0f / RAND_MAX; // 0~1均匀分布
    double Z = sqrt(-2 * log(U1))*cos(2 * M_PI * U2);
    return Z;
}

/*-------------------------------------------------------------------------------------------------------------*/

__device__ void Transform2Parameters(double *transform, double *parameters){
    parameters[0] = transform[0];
    parameters[1] = transform[1];
    parameters[2] = transform[2];
    parameters[3] = transform[3];
    parameters[4] = transform[4] - transform[6] + transform[0] * transform[6] + transform[1] * transform[7];
    parameters[5] = transform[5] - transform[7] + transform[2] * transform[6] + transform[3] * transform[7];
    parameters[6] = transform[6];
    parameters[7] = transform[7];
}

__device__ void Parameters2Transform(double *parameters, double *transform){
    transform[0] = parameters[0];
    transform[1] = parameters[1];
    transform[2] = parameters[2];
    transform[3] = parameters[3];
    transform[4] = parameters[4] + parameters[6] - parameters[0] * parameters[6] - parameters[1] * parameters[7];
    transform[5] = parameters[5] + parameters[7] - parameters[2] * parameters[6] - parameters[3] * parameters[7];
    transform[6] = parameters[6];
    transform[7] = parameters[7];
}

__device__ double ComputeMaxShift(double *d_InitialTransform, double *d_AffineTransform, double *t_AffineTransform,
                                  double offset_x, double offset_y, double maxPix_x, double maxPix_y, int x){

    double res = 0;
    double tmpMappedVoxel_x, tmpMappedVoxel_y, newMappedVoxel_x, newMappedVoxel_y,
           oldMappedVoxel_x, oldMappedVoxel_y, sampleShift;
    // 左上角的点
    tmpMappedVoxel_x = t_AffineTransform[4] + t_AffineTransform[0] * offset_x
                                            + t_AffineTransform[1] * offset_y;
    tmpMappedVoxel_y = t_AffineTransform[5] + t_AffineTransform[2] * offset_x
                                            + t_AffineTransform[3] * offset_y;
    newMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    newMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    tmpMappedVoxel_x = d_AffineTransform[4] + d_AffineTransform[0] * offset_x
                                            + d_AffineTransform[1] * offset_y;
    tmpMappedVoxel_y = d_AffineTransform[5] + d_AffineTransform[2] * offset_x
                                            + d_AffineTransform[3] * offset_y;
    oldMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    oldMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    sampleShift = sqrt((newMappedVoxel_x - oldMappedVoxel_x) * (newMappedVoxel_x - oldMappedVoxel_x) +
                       (newMappedVoxel_y - oldMappedVoxel_y) * (newMappedVoxel_y - oldMappedVoxel_y));
    if(res < sampleShift){res = sampleShift;}
    // if(x == 2 || x == 0){
    //     printf("%d:<offset_x,offset_y> [%f,%f] -> (%f,%f) / (%f,%f) sampleShift = %f\n",x,offset_x,offset_y,
    //         newMappedVoxel_x,newMappedVoxel_y,oldMappedVoxel_x,oldMappedVoxel_y,sampleShift);
    // }

    // 右上角的点
    tmpMappedVoxel_x = t_AffineTransform[4] + t_AffineTransform[0] * maxPix_x
                                            + t_AffineTransform[1] * offset_y;
    tmpMappedVoxel_y = t_AffineTransform[5] + t_AffineTransform[2] * maxPix_x
                                            + t_AffineTransform[3] * offset_y;
    newMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    newMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    tmpMappedVoxel_x = d_AffineTransform[4] + d_AffineTransform[0] * maxPix_x
                                            + d_AffineTransform[1] * offset_y;
    tmpMappedVoxel_y = d_AffineTransform[5] + d_AffineTransform[2] * maxPix_x
                                            + d_AffineTransform[3] * offset_y;
    oldMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    oldMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    sampleShift = sqrt((newMappedVoxel_x - oldMappedVoxel_x) * (newMappedVoxel_x - oldMappedVoxel_x) +
                       (newMappedVoxel_y - oldMappedVoxel_y) * (newMappedVoxel_y - oldMappedVoxel_y));
    if(res < sampleShift){res = sampleShift;}
    // if(x == 2 || x == 0){
    //     printf("%d:<maxPix_x,offset_y> [%f,%f] -> (%f,%f) / (%f,%f) sampleShift = %f\n",x,maxPix_x,offset_y,
    //         newMappedVoxel_x,newMappedVoxel_y,oldMappedVoxel_x,oldMappedVoxel_y,sampleShift);
    // }

    // 左下角的点
    tmpMappedVoxel_x = t_AffineTransform[4] + t_AffineTransform[0] * offset_x
                                            + t_AffineTransform[1] * maxPix_y;
    tmpMappedVoxel_y = t_AffineTransform[5] + t_AffineTransform[2] * offset_x
                                            + t_AffineTransform[3] * maxPix_y;
    newMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    newMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    tmpMappedVoxel_x = d_AffineTransform[4] + d_AffineTransform[0] * offset_x
                                            + d_AffineTransform[1] * maxPix_y;
    tmpMappedVoxel_y = d_AffineTransform[5] + d_AffineTransform[2] * offset_x
                                            + d_AffineTransform[3] * maxPix_y;
    oldMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    oldMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    sampleShift = sqrt((newMappedVoxel_x - oldMappedVoxel_x) * (newMappedVoxel_x - oldMappedVoxel_x) +
                       (newMappedVoxel_y - oldMappedVoxel_y) * (newMappedVoxel_y - oldMappedVoxel_y));
    if(res < sampleShift){res = sampleShift;}
    // if(x == 2 || x == 0){
    //     printf("%d:<offset_x,maxPix_y> [%f,%f] -> (%f,%f) / (%f,%f) sampleShift = %f\n",x,offset_x,maxPix_y,
    //         newMappedVoxel_x,newMappedVoxel_y,oldMappedVoxel_x,oldMappedVoxel_y,sampleShift);
    // }

    // 右下角的点
    tmpMappedVoxel_x = t_AffineTransform[4] + t_AffineTransform[0] * maxPix_x
                                            + t_AffineTransform[1] * maxPix_y;
    tmpMappedVoxel_y = t_AffineTransform[5] + t_AffineTransform[2] * maxPix_x
                                            + t_AffineTransform[3] * maxPix_y;
    newMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    newMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    tmpMappedVoxel_x = d_AffineTransform[4] + d_AffineTransform[0] * maxPix_x
                                            + d_AffineTransform[1] * maxPix_y;
    tmpMappedVoxel_y = d_AffineTransform[5] + d_AffineTransform[2] * maxPix_x
                                            + d_AffineTransform[3] * maxPix_y;
    oldMappedVoxel_x = d_InitialTransform[4] + d_InitialTransform[0] * tmpMappedVoxel_x
                                             + d_InitialTransform[1] * tmpMappedVoxel_y;
    oldMappedVoxel_y = d_InitialTransform[5] + d_InitialTransform[2] * tmpMappedVoxel_x
                                             + d_InitialTransform[3] * tmpMappedVoxel_y;
    sampleShift = sqrt((newMappedVoxel_x - oldMappedVoxel_x) * (newMappedVoxel_x - oldMappedVoxel_x) +
                       (newMappedVoxel_y - oldMappedVoxel_y) * (newMappedVoxel_y - oldMappedVoxel_y));
    if(res < sampleShift){res = sampleShift;}
    // if(x == 2 || x == 0){
    //     printf("%d:<maxPix_x,maxPix_y> [%f,%f] -> (%f,%f) / (%f,%f) sampleShift = %f\n",x,maxPix_x,maxPix_y,
    //         newMappedVoxel_x,newMappedVoxel_y,oldMappedVoxel_x,oldMappedVoxel_y,sampleShift);
    // }

    return res;
}

__device__ double EvaluateValue(int w, int h, float *d_Image, double continuousIndex_x,double continuousIndex_y){

    int basei[2];

    basei[0] = (int)continuousIndex_x;if(basei[0] < 0){basei[0] = 0;}
    double distance0 = continuousIndex_x - basei[0];

    basei[1] = (int)continuousIndex_y;if(basei[1] < 0){basei[1] = 0;}
    double distance1 = continuousIndex_y - basei[1];

    double val00 = d_Image[basei[1] * w + basei[0]];

    // if(continuousIndex_x < 163 && continuousIndex_x > 157 && continuousIndex_y < 163 && continuousIndex_y > 157){
    //     printf("    [%f,%f]:%f\n",continuousIndex_x,continuousIndex_y,val00);
    // }

    if (distance0 <= 0. && distance1 <= 0.){
        return val00;
    }
    else if (distance1 <= 0.){
        ++basei[0];
        if(basei[0] >= w){return val00;}
        double val10 = d_Image[basei[1] * w + basei[0]];
        return val00 + (val10 - val00) * distance0;
    }
    else if (distance0 <= 0.){
        ++basei[1];
        if(basei[1] >= h){return val00;}
        double val01 = d_Image[basei[1] * w + basei[0]];
        return val00 + (val01 - val00) * distance1;
    }

    ++basei[0];
    if(basei[0] >= w){
      --basei[0];
      ++basei[1];
      if(basei[1] >= h){return val00;}
      double val01 = d_Image[basei[1] * w + basei[0]];
      return val00 + (val01 - val00) * distance1;
    }

    double val10 = d_Image[basei[1] * w + basei[0]];
    double valx0 = val00 + (val10 - val00) * distance0;
    ++basei[1];
    if(basei[1] >= h){return valx0;}
    double val11 = d_Image[basei[1] * w + basei[0]];
    --basei[0];
    double val01 = d_Image[basei[1] * w + basei[0]];
    double valx1 = val01 + (val11 - val01) * distance0;
    return valx0 + (valx1 - valx0) * distance1;
}

__device__ double EvaluateValueClamped(int w, int h, float *d_Image,
                                       double continuousIndex_x, double continuousIndex_y){
    // Area samples can extend slightly beyond the center coordinate near an
    // image edge. Replicate the border instead of reading outside the image.
    continuousIndex_x = fmin(fmax(continuousIndex_x, 0.0), (double)(w - 1));
    continuousIndex_y = fmin(fmax(continuousIndex_y, 0.0), (double)(h - 1));
    return EvaluateValue(w, h, d_Image, continuousIndex_x, continuousIndex_y);
}

__device__ double EvaluateFieldValue(int w, int h, CudaFieldValueType *d_Field, double continuousIndex_x, double continuousIndex_y,
                                     int component){

    int basei[2];

    basei[0] = (int)continuousIndex_x;if(basei[0] < 0){basei[0] = 0;}
    double distance0 = continuousIndex_x - basei[0];

    basei[1] = (int)continuousIndex_y;if(basei[1] < 0){basei[1] = 0;}
    double distance1 = continuousIndex_y - basei[1];

    double val00 = d_Field[(basei[1] * w + basei[0]) * 2 + component];

    if (distance0 <= 0. && distance1 <= 0.){
        return val00;
    }
    else if (distance1 <= 0.){
        ++basei[0];
        if(basei[0] >= w){return val00;}
        double val10 = d_Field[(basei[1] * w + basei[0]) * 2 + component];
        return val00 + (val10 - val00) * distance0;
    }
    else if (distance0 <= 0.){
        ++basei[1];
        if(basei[1] >= h){return val00;}
        double val01 = d_Field[(basei[1] * w + basei[0]) * 2 + component];
        return val00 + (val01 - val00) * distance1;
    }

    ++basei[0];
    if(basei[0] >= w){
      --basei[0];
      ++basei[1];
      if(basei[1] >= h){return val00;}
      double val01 = d_Field[(basei[1] * w + basei[0]) * 2 + component];
      return val00 + (val01 - val00) * distance1;
    }

    double val10 = d_Field[(basei[1] * w + basei[0]) * 2 + component];
    double valx0 = val00 + (val10 - val00) * distance0;
    ++basei[1];
    if(basei[1] >= h){return valx0;}
    double val11 = d_Field[(basei[1] * w + basei[0]) * 2 + component];
    --basei[0];
    double val01 = d_Field[(basei[1] * w + basei[0]) * 2 + component];
    double valx1 = val01 + (val11 - val01) * distance0;
    return valx0 + (valx1 - valx0) * distance1;
}

__device__ void ApplyInverseLinearTransform(double *d_Transform, double x, double y, double &true_x, double &true_y){
    double det = d_Transform[0] * d_Transform[3] - d_Transform[1] * d_Transform[2];
    double shifted_x = x - d_Transform[4];
    double shifted_y = y - d_Transform[5];

    true_x = ( d_Transform[3] * shifted_x - d_Transform[1] * shifted_y) / det;
    true_y = (-d_Transform[2] * shifted_x + d_Transform[0] * shifted_y) / det;
}

__device__ bool MapFixedToMovingSource(int w, int h, double outputX, double outputY,
                                       double *d_InitialTransform, double *d_AffineTransform,
                                       CudaFieldValueType *d_OutputInverseFieldFinal,
                                       double &sourceX, double &sourceY){
    double initialInverseX;
    double initialInverseY;
    ApplyInverseLinearTransform(d_InitialTransform, outputX, outputY, initialInverseX, initialInverseY);

    double affineInverseX;
    double affineInverseY;
    ApplyInverseLinearTransform(d_AffineTransform, initialInverseX, initialInverseY,
                                affineInverseX, affineInverseY);
    if(affineInverseX < -0.5 || affineInverseX > w - 0.5 ||
       affineInverseY < -0.5 || affineInverseY > h - 0.5){
        return false;
    }

    sourceX = affineInverseX
            + EvaluateFieldValue(w, h, d_OutputInverseFieldFinal, affineInverseX, affineInverseY, 0);
    sourceY = affineInverseY
            + EvaluateFieldValue(w, h, d_OutputInverseFieldFinal, affineInverseX, affineInverseY, 1);
    return sourceX >= -0.5 && sourceX <= w - 0.5 &&
           sourceY >= -0.5 && sourceY <= h - 0.5;
}

__device__ double ClampValue(double value, double lower, double upper){
    if(value < lower){return lower;}
    if(value > upper){return upper;}
    return value;
}

__device__ double RobustDiff(double diff, double robustThreshold){
    if(robustThreshold <= 0){
        return diff;
    }
    return ClampValue(diff, -robustThreshold, robustThreshold);
}

__device__ void EvaluateGradient(int w, int h, float *d_Image, double continuousIndex_x, double continuousIndex_y,
                                 double &gradient_x, double &gradient_y){

    gradient_x = EvaluateValue(w, h, d_Image, continuousIndex_x + 0.5, continuousIndex_y) -
                 EvaluateValue(w, h, d_Image, continuousIndex_x - 0.5, continuousIndex_y);

    gradient_y = EvaluateValue(w, h, d_Image, continuousIndex_x, continuousIndex_y + 0.5) -
                 EvaluateValue(w, h, d_Image, continuousIndex_x, continuousIndex_y - 0.5);

    if(continuousIndex_x < 0 || continuousIndex_x >= w - 1){
        gradient_x = 0;
    }
    if(continuousIndex_y < 0 || continuousIndex_y >= h - 1){
        gradient_y = 0;
    }
}

/*-------------------------------------------------------------------------------------------------------------*/

__global__ void cudaEstimateScalesKermal(double *d_InitialTransform, double *d_AffineTransform, double offset_x,
                                         double offset_y, double maxPix_x, double maxPix_y, double *d_Scales){

    int x = threadIdx.x;
    double AffineParameters[8], t_AffineTransform[8];
    Transform2Parameters(d_AffineTransform, AffineParameters);
    AffineParameters[x] += 0.01;
    Parameters2Transform(AffineParameters, t_AffineTransform);
    // if(x == 0 || x == 2){
    //     printf("d_AffineTransform:[%f,%f,%f,%f,%f,%f,%f,%f] %d\n",d_AffineTransform[0],d_AffineTransform[1],
    //         d_AffineTransform[2],d_AffineTransform[3],d_AffineTransform[4],
    //         d_AffineTransform[5],d_AffineTransform[6],d_AffineTransform[7],x);
    //     printf("AffineParameters:[%f,%f,%f,%f,%f,%f,%f,%f] %d\n",AffineParameters[0],AffineParameters[1],
    //         AffineParameters[2],AffineParameters[3],AffineParameters[4],
    //         AffineParameters[5],AffineParameters[6],AffineParameters[7],x);
    //     printf("t_AffineTransform:[%f,%f,%f,%f,%f,%f,%f,%f] %d\n",t_AffineTransform[0],t_AffineTransform[1],
    //         t_AffineTransform[2],t_AffineTransform[3],t_AffineTransform[4],
    //         t_AffineTransform[5],t_AffineTransform[6],t_AffineTransform[7],x);
    // }

    d_Scales[x] = ComputeMaxShift(d_InitialTransform, d_AffineTransform, t_AffineTransform,
                                  offset_x, offset_y, maxPix_x, maxPix_y, x);

    d_Scales[x] *= d_Scales[x] * 10000;
}

__global__ void cudaEstimateLearningRateKernal(double *d_InitialTransform, double *d_AffineTransform, double *d_Gradient,
                                               double offset_x, double offset_y, double maxPix_x, double maxPix_y,
                                               double *d_LearningRate){

    // A stationary gradient has no learning rate to estimate.  In particular,
    // avoid 0.01 / 0 followed by 0 * infinity corrupting the affine transform.
    *d_LearningRate = 0.0;
    double maxStep = 0.0;
    for(int parameter = 0; parameter < 6; ++parameter){
        if(!isfinite(d_Gradient[parameter])){return;}
        maxStep = fmax(maxStep, fabs(d_Gradient[parameter]));
    }
    if(maxStep == 0.0){return;}

    double factor = 0.01 / maxStep;
    if(!isfinite(factor) || factor <= 0.0){return;}
    double AffineParameters[8], t_AffineTransform[8];
    Transform2Parameters(d_AffineTransform, AffineParameters);
    AffineParameters[0] += d_Gradient[0] * factor;
    AffineParameters[1] += d_Gradient[1] * factor;
    AffineParameters[2] += d_Gradient[2] * factor;
    AffineParameters[3] += d_Gradient[3] * factor;
    AffineParameters[4] += d_Gradient[4] * factor;
    AffineParameters[5] += d_Gradient[5] * factor;
    Parameters2Transform(AffineParameters, t_AffineTransform);
    double stepScale = ComputeMaxShift(d_InitialTransform, d_AffineTransform, t_AffineTransform,
                                       offset_x, offset_y, maxPix_x, maxPix_y, 1) / factor;
    if(!isfinite(stepScale) || stepScale <= 0.0){return;}
    double learningRate = 0.25 / stepScale;
    if(isfinite(learningRate) && learningRate > 0.0){
        *d_LearningRate = learningRate;
    }
}

__global__ void cudaCalculateAffineKernal(int w, int h, int spacing, float *d_FixedImage, float *d_MovingImage,
                                          double *d_InitialTransform, double *d_AffineTransform, double *d_VirtualPointSet,
                                          bool ComputeDerivative, int NumberOfDomainPoints, int pointsInWidth,
                                          int pointsInHeightCeil, CudaAffineMetricValueType *d_DerivativeResult,
                                          CudaAffineMetricValueType *d_ValueResult){

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= NumberOfDomainPoints){return;}

    double virtualPoint_x = d_VirtualPointSet[index * 2 + 0];
    double virtualPoint_y = d_VirtualPointSet[index * 2 + 1];

    bool pointIsValid = true;

    if(virtualPoint_x < -0.5 || virtualPoint_x >= w - 0.5 || virtualPoint_y < -0.5 || virtualPoint_y >= h - 0.5){
        pointIsValid = false;
    }

    double mappedFixedPixelValue = 0;

    if(pointIsValid){
        mappedFixedPixelValue = EvaluateValue(w, h, d_FixedImage, virtualPoint_x, virtualPoint_y);
    }

    double mappedMovingPoint_x_temp = d_AffineTransform[4] + d_AffineTransform[0] * virtualPoint_x
                                                           + d_AffineTransform[1] * virtualPoint_y;
    double mappedMovingPoint_y_temp = d_AffineTransform[5] + d_AffineTransform[2] * virtualPoint_x
                                                           + d_AffineTransform[3] * virtualPoint_y;

    double mappedMovingPoint_x = d_InitialTransform[4] + d_InitialTransform[0] * mappedMovingPoint_x_temp
                                                       + d_InitialTransform[1] * mappedMovingPoint_y_temp;
    double mappedMovingPoint_y = d_InitialTransform[5] + d_InitialTransform[2] * mappedMovingPoint_x_temp
                                                       + d_InitialTransform[3] * mappedMovingPoint_y_temp;

    if(mappedMovingPoint_x < -0.5 || mappedMovingPoint_x >= w - 0.5 ||
       mappedMovingPoint_y < -0.5 || mappedMovingPoint_y >= h - 0.5){
        pointIsValid = false;
    }

    double mappedMovingPixelValue = 0;

    double mappedMovingPixelGradient_x = 0.0;
    double mappedMovingPixelGradient_y = 0.0;

    if(pointIsValid){
        mappedMovingPixelValue = EvaluateValue(w, h, d_MovingImage, mappedMovingPoint_x, mappedMovingPoint_y);
        if(ComputeDerivative){
            EvaluateGradient(w, h, d_MovingImage, mappedMovingPoint_x, mappedMovingPoint_y,
                             mappedMovingPixelGradient_x, mappedMovingPixelGradient_y);
        }
    }

//     if(virtualPoint_x == 158.5 && virtualPoint_y == 157.5 || index == 256){
//         printf("mappedFixedPixelValue:%f [%f,%f]; mappedMovingPixelValue:%f [%f,%f]; mappedMovingPixelGradient:[%f,%f]\n",
//             mappedFixedPixelValue,virtualPoint_x,virtualPoint_y,
//             mappedMovingPixelValue,mappedMovingPoint_x,mappedMovingPoint_y,
//             mappedMovingPixelGradient_x,mappedMovingPixelGradient_y);
//     }

    if(pointIsValid){
        double v_x = virtualPoint_x - d_AffineTransform[6];
        double v_y = virtualPoint_y - d_AffineTransform[7];

        double diff = mappedFixedPixelValue - mappedMovingPixelValue;

        if(ComputeDerivative){
            double jacobian_0_0 = d_InitialTransform[0] * v_x;
            double jacobian_0_1 = d_InitialTransform[0] * v_y;
            double jacobian_0_2 = d_InitialTransform[1] * v_x;
            double jacobian_0_3 = d_InitialTransform[1] * v_y;
            double jacobian_0_4 = d_InitialTransform[0];
            double jacobian_0_5 = d_InitialTransform[1];
            double jacobian_1_0 = d_InitialTransform[2] * v_x;
            double jacobian_1_1 = d_InitialTransform[2] * v_y;
            double jacobian_1_2 = d_InitialTransform[3] * v_x;
            double jacobian_1_3 = d_InitialTransform[3] * v_y;
            double jacobian_1_4 = d_InitialTransform[2];
            double jacobian_1_5 = d_InitialTransform[3];

            d_DerivativeResult[index * 6 + 0] = static_cast<CudaAffineMetricValueType>(
                2.0 * diff * jacobian_0_0 * mappedMovingPixelGradient_x
              + 2.0 * diff * jacobian_1_0 * mappedMovingPixelGradient_y);
            d_DerivativeResult[index * 6 + 1] = static_cast<CudaAffineMetricValueType>(
                2.0 * diff * jacobian_0_1 * mappedMovingPixelGradient_x
              + 2.0 * diff * jacobian_1_1 * mappedMovingPixelGradient_y);
            d_DerivativeResult[index * 6 + 2] = static_cast<CudaAffineMetricValueType>(
                2.0 * diff * jacobian_0_2 * mappedMovingPixelGradient_x
              + 2.0 * diff * jacobian_1_2 * mappedMovingPixelGradient_y);
            d_DerivativeResult[index * 6 + 3] = static_cast<CudaAffineMetricValueType>(
                2.0 * diff * jacobian_0_3 * mappedMovingPixelGradient_x
              + 2.0 * diff * jacobian_1_3 * mappedMovingPixelGradient_y);
            d_DerivativeResult[index * 6 + 4] = static_cast<CudaAffineMetricValueType>(
                2.0 * diff * jacobian_0_4 * mappedMovingPixelGradient_x
              + 2.0 * diff * jacobian_1_4 * mappedMovingPixelGradient_y);
            d_DerivativeResult[index * 6 + 5] = static_cast<CudaAffineMetricValueType>(
                2.0 * diff * jacobian_0_5 * mappedMovingPixelGradient_x
              + 2.0 * diff * jacobian_1_5 * mappedMovingPixelGradient_y);
        }
        d_ValueResult[index] = static_cast<CudaAffineMetricValueType>(diff * diff);
    }
    else{
        if(ComputeDerivative){
            d_DerivativeResult[index * 6 + 0] = 0;
            d_DerivativeResult[index * 6 + 1] = 0;
            d_DerivativeResult[index * 6 + 2] = 0;
            d_DerivativeResult[index * 6 + 3] = 0;
            d_DerivativeResult[index * 6 + 4] = 0;
            d_DerivativeResult[index * 6 + 5] = 0;
        }
        d_ValueResult[index] = static_cast<CudaAffineMetricValueType>(-1);
    }
}

__global__ void cudaGetRowGradientKernal(int w,int h, int spacing, CudaAffineMetricValueType *d_DerivativeResult,
                                         CudaAffineMetricValueType *d_ValueResult,
                                         int NumberOfDomainPoints, int pointsInWidth, int pointsInHeightCeil, double *t_Gradient,
                                         CudaAffineMetricValueType *t_Value, CudaAffineMetricValueType *t_ValidPointCount){

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if(row >= pointsInHeightCeil){return;}

    // The affine process stage was previously one thread scanning a whole row.
    // Use one block per sampled row and reduce within shared memory to cut the
    // serial reduction cost without changing the optimizer math.
    __shared__ double sGradient0[256];
    __shared__ double sGradient1[256];
    __shared__ double sGradient2[256];
    __shared__ double sGradient3[256];
    __shared__ double sGradient4[256];
    __shared__ double sGradient5[256];
    __shared__ double sValue[256];
    __shared__ int sValidPointCount[256];

    double gradient0 = 0.0;
    double gradient1 = 0.0;
    double gradient2 = 0.0;
    double gradient3 = 0.0;
    double gradient4 = 0.0;
    double gradient5 = 0.0;
    double value = 0.0;
    int validPointCount = 0;

    for(int i = tid; i < pointsInWidth; i += blockDim.x){
        int trueIndex = row * pointsInWidth + i;
        if(trueIndex >= NumberOfDomainPoints){break;}
        if(d_ValueResult[trueIndex] != -1){
            gradient0 += d_DerivativeResult[trueIndex * 6 + 0];
            gradient1 += d_DerivativeResult[trueIndex * 6 + 1];
            gradient2 += d_DerivativeResult[trueIndex * 6 + 2];
            gradient3 += d_DerivativeResult[trueIndex * 6 + 3];
            gradient4 += d_DerivativeResult[trueIndex * 6 + 4];
            gradient5 += d_DerivativeResult[trueIndex * 6 + 5];
            value += d_ValueResult[trueIndex];
            validPointCount++;
        }
    }

    sGradient0[tid] = gradient0;
    sGradient1[tid] = gradient1;
    sGradient2[tid] = gradient2;
    sGradient3[tid] = gradient3;
    sGradient4[tid] = gradient4;
    sGradient5[tid] = gradient5;
    sValue[tid] = value;
    sValidPointCount[tid] = validPointCount;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sGradient0[tid] += sGradient0[tid + stride];
            sGradient1[tid] += sGradient1[tid + stride];
            sGradient2[tid] += sGradient2[tid + stride];
            sGradient3[tid] += sGradient3[tid + stride];
            sGradient4[tid] += sGradient4[tid + stride];
            sGradient5[tid] += sGradient5[tid + stride];
            sValue[tid] += sValue[tid + stride];
            sValidPointCount[tid] += sValidPointCount[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        t_Gradient[row * 6 + 0] = sGradient0[0];
        t_Gradient[row * 6 + 1] = sGradient1[0];
        t_Gradient[row * 6 + 2] = sGradient2[0];
        t_Gradient[row * 6 + 3] = sGradient3[0];
        t_Gradient[row * 6 + 4] = sGradient4[0];
        t_Gradient[row * 6 + 5] = sGradient5[0];
        t_Value[row] = static_cast<CudaAffineMetricValueType>(sValue[0]);
        t_ValidPointCount[row] = static_cast<CudaAffineMetricValueType>(sValidPointCount[0]);
    }
}

__global__ void cudaGetColGradientKernal(int w, int h, int spacing, double *t_Gradient,
                                         CudaAffineMetricValueType *t_Value,
                                         CudaAffineMetricValueType *t_ValidPointCount, double *d_Scales, int NumberOfDomainPoints,
                                         int pointsInWidth, int pointsInHeightCeil, int iteration, int minIteration,
                                         double *d_Gradient, double *d_Value, double *d_EnergyValues, double *d_TotalEnergy){
    //
    int validPointCount = 0;
    d_Gradient[0] = 0;  d_Gradient[1] = 0;  d_Gradient[2] = 0;
    d_Gradient[3] = 0;  d_Gradient[4] = 0;  d_Gradient[5] = 0;
    *d_Value = 0;

    for(int i = 0; i < pointsInHeightCeil; i++){
        d_Gradient[0] += t_Gradient[i * 6 + 0];
        d_Gradient[1] += t_Gradient[i * 6 + 1];
        d_Gradient[2] += t_Gradient[i * 6 + 2];
        d_Gradient[3] += t_Gradient[i * 6 + 3];
        d_Gradient[4] += t_Gradient[i * 6 + 4];
        d_Gradient[5] += t_Gradient[i * 6 + 5];
        *d_Value += t_Value[i];
        // printf("t_Value[%d]:%f\n",i,t_Value[i]);
        validPointCount += t_ValidPointCount[i];
    }

    d_Gradient[0] /= validPointCount;
    d_Gradient[1] /= validPointCount;
    d_Gradient[2] /= validPointCount;
    d_Gradient[3] /= validPointCount;
    d_Gradient[4] /= validPointCount;
    d_Gradient[5] /= validPointCount;

    *d_Value /= validPointCount;

//     printf("d_Value:%f validPointCount:%d d_Gradient:[%f,%f,%f,%f,%f,%f]\n",*d_Value,validPointCount,
//         d_Gradient[0],d_Gradient[1],d_Gradient[2],d_Gradient[3],d_Gradient[4],d_Gradient[5]);
//     printf("d_Scales:[%f,%f,%f,%f,%f,%f]\n",d_Scales[0],d_Scales[1],d_Scales[2],d_Scales[3],d_Scales[4],d_Scales[5]);

    d_Gradient[0] /= d_Scales[0];
    d_Gradient[1] /= d_Scales[1];
    d_Gradient[2] /= d_Scales[2];
    d_Gradient[3] /= d_Scales[3];
    d_Gradient[4] /= d_Scales[4];
    d_Gradient[5] /= d_Scales[5];

    d_EnergyValues[iteration % minIteration] = *d_Value;
    *d_TotalEnergy += *d_Value;
}

__global__ void cudaGetRowValueKernal(int w, int h, int spacing, CudaAffineMetricValueType *d_ValueResult, int NumberOfDomainPoints,
                                      int pointsInWidth, int pointsInHeightCeil, CudaAffineMetricValueType *t_Value,
                                      CudaAffineMetricValueType *t_ValidPointCount){

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if(row >= pointsInHeightCeil){return;}

    __shared__ double sValue[256];
    __shared__ int sValidPointCount[256];

    double value = 0.0;
    int validPointCount = 0;
    for(int i = tid; i < pointsInWidth; i += blockDim.x){
        int trueIndex = row * pointsInWidth + i;
        if(trueIndex >= NumberOfDomainPoints){break;}
        if(d_ValueResult[trueIndex] != -1){
            value += d_ValueResult[trueIndex];
            validPointCount++;
        }
    }

    sValue[tid] = value;
    sValidPointCount[tid] = validPointCount;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            sValue[tid] += sValue[tid + stride];
            sValidPointCount[tid] += sValidPointCount[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        t_Value[row] = static_cast<CudaAffineMetricValueType>(sValue[0]);
        t_ValidPointCount[row] = static_cast<CudaAffineMetricValueType>(sValidPointCount[0]);
    }
}

__global__ void cudaGetColValueKernal(int w, int h, int spacing, CudaAffineMetricValueType *t_Value,
                                      CudaAffineMetricValueType *t_ValidPointCount,
                                      int NumberOfDomainPoints, int pointsInWidth, int pointsInHeightCeil, double *d_Value){

    int validPointCount = 0;
    *d_Value = 0;

    for(int i = 0; i < pointsInHeightCeil; i++){
        *d_Value += t_Value[i];
        validPointCount += t_ValidPointCount[i];
    }
    *d_Value /= validPointCount;

//     printf("d_Value:%f validPointCount:%d\n",*d_Value,validPointCount);
}

__global__ void cudaResetGradientKernal(double *d_Gradient, double *d_LearningRate){
    int x = threadIdx.x;
    d_Gradient[x] = d_Gradient[x] / (*d_LearningRate);
}

__global__ void cudaResetTransformKernal(double *d_AffineTransform, double *d_Gradient){
    double AffineParameters[8];
    Transform2Parameters(d_AffineTransform, AffineParameters);
    AffineParameters[0] -= d_Gradient[0];
    AffineParameters[1] -= d_Gradient[1];
    AffineParameters[2] -= d_Gradient[2];
    AffineParameters[3] -= d_Gradient[3];
    AffineParameters[4] -= d_Gradient[4];
    AffineParameters[5] -= d_Gradient[5];
    Parameters2Transform(AffineParameters, d_AffineTransform);
}

__global__ void cudaModifyGradientKernal(double *d_Gradient, double *d_LearningRate){
    int x = threadIdx.x;
    // A rejected/stationary step must remain zero even if the input gradient
    // was non-finite; multiplying NaN by zero would otherwise preserve NaN.
    if(!isfinite(*d_LearningRate) || *d_LearningRate <= 0.0){
        d_Gradient[x] = 0.0;
        return;
    }
    d_Gradient[x] = d_Gradient[x] * (*d_LearningRate);
}

__global__ void cudaUpdateTransformKernal(double *d_AffineTransform, double *d_Gradient){
    double AffineParameters[8];
    Transform2Parameters(d_AffineTransform, AffineParameters);
    AffineParameters[0] += d_Gradient[0];
    AffineParameters[1] += d_Gradient[1];
    AffineParameters[2] += d_Gradient[2];
    AffineParameters[3] += d_Gradient[3];
    AffineParameters[4] += d_Gradient[4];
    AffineParameters[5] += d_Gradient[5];
    Parameters2Transform(AffineParameters, d_AffineTransform);
}

__global__ void cudaGetLatticeKernal(double *d_EnergyValues, double *d_TotalEnergy, int iteration, int minIteration,
                                     double *d_DeltaLattice, double *d_OmegaLattice){

    int n = (iteration + 1 + threadIdx.x) % minIteration;
    int index = threadIdx.x;

    double p = (double)index / (minIteration - 1);
    // if(p == 1){p = 0.9999;}

    double nbhd[2]; // neighborhoodWeightImage
    double w2S = 0; // w2Sum
    double B;

    B = 1 - abs(p - 0);
    nbhd[0] = B;
    w2S += B * B;

    B = 1 - abs(p - 1);
    nbhd[1] = B;
    w2S += B * B;

    // printf("No.%d (index=%d):%f p:%f nbhd:[%f,%f]\n",n,index,d_EnergyValues[n],p,nbhd[0],nbhd[1]);

    d_OmegaLattice[n * 2 + 0] = nbhd[0] * nbhd[0];
    d_OmegaLattice[n * 2 + 1] = nbhd[1] * nbhd[1];
    d_DeltaLattice[n * 2 + 0] = (d_EnergyValues[n] / (*d_TotalEnergy)) * (nbhd[0] * nbhd[0] * nbhd[0] / w2S);
    d_DeltaLattice[n * 2 + 1] = (d_EnergyValues[n] / (*d_TotalEnergy)) * (nbhd[1] * nbhd[1] * nbhd[1] / w2S);

}

__global__ void cudaGetConvergenceValueKernal(double *d_DeltaLattice, double *d_OmegaLattice, double *d_ConvergenceValue){

    double delta_0 = d_DeltaLattice[0] + d_DeltaLattice[2] + d_DeltaLattice[4] + d_DeltaLattice[6] +
                     d_DeltaLattice[8] + d_DeltaLattice[10] + d_DeltaLattice[12] + d_DeltaLattice[14] +
                     d_DeltaLattice[16] + d_DeltaLattice[18];
    double delta_1 = d_DeltaLattice[1] + d_DeltaLattice[3] + d_DeltaLattice[5] + d_DeltaLattice[7] +
                     d_DeltaLattice[9] + d_DeltaLattice[11] + d_DeltaLattice[13] + d_DeltaLattice[15] +
                     d_DeltaLattice[17] + d_DeltaLattice[19];
    double omega_0 = d_OmegaLattice[0] + d_OmegaLattice[2] + d_OmegaLattice[4] + d_OmegaLattice[6] +
                     d_OmegaLattice[8] + d_OmegaLattice[10] + d_OmegaLattice[12] + d_OmegaLattice[14] +
                     d_OmegaLattice[16] + d_OmegaLattice[18];
    double omega_1 = d_OmegaLattice[1] + d_OmegaLattice[3] + d_OmegaLattice[5] + d_OmegaLattice[7] +
                     d_OmegaLattice[9] + d_OmegaLattice[11] + d_OmegaLattice[13] + d_OmegaLattice[15] +
                     d_OmegaLattice[17] + d_OmegaLattice[19];
    double phi_0,phi_1;

    if(delta_0 == 0 || omega_0 == 0){phi_0 = 0;}else{phi_0 = delta_0 / omega_0;}
    if(delta_1 == 0 || omega_1 == 0){phi_1 = 0;}else{phi_1 = delta_1 / omega_1;}
    *d_ConvergenceValue = phi_0 - phi_1;
    // printf("delta:[%f,%f]; omega:[%f,%f]; d_PhiLattice:[%f,%f]; d_ConvergenceValue:%f\n",
    //     delta_0, delta_1, omega_0, omega_1, phi_0, phi_1, *d_ConvergenceValue);
}

__global__ void cudaSetGoldenSectionKernal(double *d_LearningRate, double *d_Gradient,
                                          double *a, double *b, double *c){
    // Validate every iteration on device, including later iterations that reuse
    // the previous learning rate. No additional host transfer is required.
    bool hasGradient = false;
    bool validGradient = true;
    for(int parameter = 0; parameter < 6; ++parameter){
        hasGradient = hasGradient || d_Gradient[parameter] != 0.0;
        validGradient = validGradient && isfinite(d_Gradient[parameter]);
    }
    if(!hasGradient || !validGradient || !isfinite(*d_LearningRate) ||
       *d_LearningRate <= 0.0 || !isfinite(2.0 * (*d_LearningRate))){
        *d_LearningRate = 0.0;
        for(int parameter = 0; parameter < 6; ++parameter){d_Gradient[parameter] = 0.0;}
        *a = *b = *c = 0.0;
        return;
    }
    *a = 0 * (*d_LearningRate);
    *b = 1 * (*d_LearningRate);
    *c = 2 * (*d_LearningRate);
}

__global__ void cudaFindGoldenSectionKernal(double *a, double *b, double *c, double *x, double *metricb, double *metricx){
//     printf("metricx:%.10f metricb:%.10f\n",*metricx,*metricb);
    if (*metricx < *metricb)
    {
        if (*c - *b > *b - *a){
            *a = *b;    *b = *x;    *c = *c;
            *metricb = *metricx;
        }
        else{
            *c = *b;    *b = *x;    *a = *a;
            *metricb = *metricx;
        }
    }
    else
    {
        if (*c - *b > *b - *a){
            *c = *x;    *b = *b;    *a = *a;
            *metricb = *metricb;
        }
        else{
            *a = *x;    *b = *b;    *c = *c;
            *metricb = *metricb;
        }
    }
}

__global__ void cudaCalcGoldenSectionKernal(int iteration, double *a, double *b, double *c, double *x, double *d_LearningRate,
                                            double *baseLearningRate, double *flag){
    //
//     printf("a = %f b = %f c = %f\n",*a,*b,*c);

    *flag = 0;
    // A collapsed interval represents a zero/rejected step. Return before any
    // trial update or gradient restoration can divide by a zero learning rate.
    if(!isfinite(*d_LearningRate) || *d_LearningRate <= 0.0 ||
       !isfinite(*a) || !isfinite(*b) || !isfinite(*c) || *c <= *a){
        *d_LearningRate = 0.0;
        *flag = 1;
        return;
    }
    if(iteration >= 20){
        *d_LearningRate = (*c + *a) / 2;
        *flag = 1;
    }
    else{
        if(*c - *b > *b - *a){
            *x = *b + 0.381966 * (*c - *b);
        }
        else{
            *x = *b - 0.381966 * (*b - *a);
        }
        if(abs(*c - *a) < 0.2 * (abs(*b) + abs(*x))){
            *d_LearningRate = (*c + *a) / 2;
            *flag = 1;
        }
        else{
            *baseLearningRate = *d_LearningRate;
            *d_LearningRate = *x;
        }
    }
}

/*-------------------------------------------------------------------------------------------------------------*/

__global__ void cudaSetParameterKernal(double *a, double *b){
    *a = *b;
}

__global__ void cudaPrintTransformKernal(double *d_Transform){
//     printf("current transform:[%f,%f,%f,%f][%f,%f]\n",
//         d_Transform[0],d_Transform[1],d_Transform[2],
//         d_Transform[3],d_Transform[4],d_Transform[5]);
}

/*-------------------------------------------------------------------------------------------------------------*/

__global__ void cudaResampleFixedImageKernal(int w, int h, int spacing, float *d_FixedImage, CudaFieldValueType *d_FixedInverseField,
                                             double offset_x, double offset_y, float *d_ResampledFixedImage){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outW = w / spacing;
    int outH = h / spacing;
    if(x >= outW || y >= outH){return;}

    double true_x = (x * spacing + offset_x) + d_FixedInverseField[(y * outW + x) * 2 + 0];
    double true_y = (y * spacing + offset_y) + d_FixedInverseField[(y * outW + x) * 2 + 1];

    if(true_x < -0.5 || true_x > w - 0.5 || true_y < -0.5 || true_y > h - 0.5){
        d_ResampledFixedImage[y * outW + x] = 0;
    }
    else{
        d_ResampledFixedImage[y * outW + x] = (float)EvaluateValue(w, h, d_FixedImage, true_x, true_y);
    }

//     if(x == 32 && y == 64){
//         printf("resampledFixedImage[%f,%f]:%f\n",true_x,true_y,d_ResampledFixedImage[y * (w / spacing) + x]);
//     }

}

__global__ void cudaResampleMovingImageKernal(int w, int h, int spacing, float *d_MovingImage, double *d_InitialTransform,
                                              double *d_AffineTransform, CudaFieldValueType *d_MovingInverseField, double offset_x,
                                              double offset_y, float *d_ResampledMovingImage){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outW = w / spacing;
    int outH = h / spacing;
    if(x >= outW || y >= outH){return;}

    double temp_x = (x * spacing + offset_x) + d_MovingInverseField[(y * outW + x) * 2 + 0];
    double temp_y = (y * spacing + offset_y) + d_MovingInverseField[(y * outW + x) * 2 + 1];

    double temp_x_2 = d_AffineTransform[4] + d_AffineTransform[0] * temp_x
                                           + d_AffineTransform[1] * temp_y;
    double temp_y_2 = d_AffineTransform[5] + d_AffineTransform[2] * temp_x
                                           + d_AffineTransform[3] * temp_y;

    double true_x = d_InitialTransform[4] + d_InitialTransform[0] * temp_x_2
                                          + d_InitialTransform[1] * temp_y_2;
    double true_y = d_InitialTransform[5] + d_InitialTransform[2] * temp_x_2
                                          + d_InitialTransform[3] * temp_y_2;

    if(true_x < -0.5 || true_x > w - 0.5 || true_y < -0.5 || true_y > h - 0.5){
        d_ResampledMovingImage[y * outW + x] = 0;
    }
    else{
        d_ResampledMovingImage[y * outW + x] = (float)EvaluateValue(w, h, d_MovingImage, true_x, true_y);
    }

//     if(x == 32 && y == 64){
//         printf("resampledMovingImage[%f,%f]:%f\n",true_x,true_y,d_ResampledMovingImage[y * (w / spacing) + x]);
//     }
}

__global__ void cudaCalculateSynKernal(int w, int h, int spacing, float *d_FixedImage, float *d_MovingImage,
                                       CudaFieldValueType *d_FixedDerivative, CudaFieldValueType *d_MovingDerivative,
                                       double *d_ValueResult, double robustThreshold){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute local statistics after both images have been resampled into the
    // current SyN coordinate system.  This is a local-NCC update rather than
    // pre-normalized SSD, so illumination changes do not turn text edges into
    // false deformation forces.
    constexpr int radius = 4;
    constexpr int windowSize = radius * 2 + 1;
    const int tileWidth = blockDim.x + radius * 2;
    const int tileHeight = blockDim.y + radius * 2;
    const int tilePixels = tileWidth * tileHeight;
    extern __shared__ float sharedImages[];
    float *fixedTile = sharedImages;
    float *movingTile = sharedImages + tilePixels;

    const int localLinear = threadIdx.y * blockDim.x + threadIdx.x;
    const int blockThreads = blockDim.x * blockDim.y;
    for(int tileIndex = localLinear; tileIndex < tilePixels; tileIndex += blockThreads){
        int tileX = tileIndex % tileWidth;
        int tileY = tileIndex / tileWidth;
        int imageX = (int)(blockIdx.x * blockDim.x) + tileX - radius;
        int imageY = (int)(blockIdx.y * blockDim.y) + tileY - radius;
        imageX = max(0, min(imageX, w - 1));
        imageY = max(0, min(imageY, h - 1));
        fixedTile[tileIndex] = d_FixedImage[imageY * w + imageX];
        movingTile[tileIndex] = d_MovingImage[imageY * w + imageX];
    }
    __syncthreads();

    if(x >= w || y >= h){return;}

    double fixedSum = 0.0;
    double movingSum = 0.0;
    double fixedSquareSum = 0.0;
    double movingSquareSum = 0.0;
    double crossSum = 0.0;
    for(int wy = 0; wy < windowSize; ++wy){
        for(int wx = 0; wx < windowSize; ++wx){
            int tileIndex = (threadIdx.y + wy) * tileWidth + threadIdx.x + wx;
            double fixedValue = fixedTile[tileIndex];
            double movingValue = movingTile[tileIndex];
            fixedSum += fixedValue;
            movingSum += movingValue;
            fixedSquareSum += fixedValue * fixedValue;
            movingSquareSum += movingValue * movingValue;
            crossSum += fixedValue * movingValue;
        }
    }

    constexpr double sampleCount = (double)(windowSize * windowSize);
    double fixedMean = fixedSum / sampleCount;
    double movingMean = movingSum / sampleCount;
    double fixedVariance = max(fixedSquareSum - fixedSum * fixedMean, 0.0);
    double movingVariance = max(movingSquareSum - movingSum * movingMean, 0.0);
    double covariance = crossSum - fixedSum * movingMean;

    double fixedCenter = fixedTile[(threadIdx.y + radius) * tileWidth + threadIdx.x + radius] - fixedMean;
    double movingCenter = movingTile[(threadIdx.y + radius) * tileWidth + threadIdx.x + radius] - movingMean;

    double mappedFixedPixelGradient_x = 0.0;
    double mappedFixedPixelGradient_y = 0.0;

    double mappedMovingPixelGradient_x = 0.0;
    double mappedMovingPixelGradient_y = 0.0;

    EvaluateGradient(w, h, d_FixedImage, x, y, mappedFixedPixelGradient_x, mappedFixedPixelGradient_y);
    EvaluateGradient(w, h, d_MovingImage, x, y, mappedMovingPixelGradient_x, mappedMovingPixelGradient_y);

    mappedFixedPixelGradient_x /= spacing;
    mappedFixedPixelGradient_y /= spacing;
    mappedMovingPixelGradient_x /= spacing;
    mappedMovingPixelGradient_y /= spacing;

    // Flat windows carry no reliable geometric information.  Suppressing their
    // update also avoids amplifying camera noise in plain background regions.
    constexpr double varianceEpsilon = 1e-6;
    if(fixedVariance <= varianceEpsilon || movingVariance <= varianceEpsilon){
        d_ValueResult[y * w + x] = 0.0;
        d_FixedDerivative[(y * w + x) * 2 + 0] = 0.0f;
        d_FixedDerivative[(y * w + x) * 2 + 1] = 0.0f;
        d_MovingDerivative[(y * w + x) * 2 + 0] = 0.0f;
        d_MovingDerivative[(y * w + x) * 2 + 1] = 0.0f;
        return;
    }

    double inverseNorm = rsqrt(fixedVariance * movingVariance + varianceEpsilon);
    double correlation = ClampValue(covariance * inverseNorm, -1.0, 1.0);
    double fixedCorrelationDerivative = movingCenter * inverseNorm
                                      - correlation * fixedCenter / fixedVariance;
    double movingCorrelationDerivative = fixedCenter * inverseNorm
                                       - correlation * movingCenter / movingVariance;

    // The negative metric derivative is used as the update direction for both
    // halves of the symmetric transform, matching the existing SyN convention.
    d_ValueResult[y * w + x] = 1.0 - correlation;
    d_FixedDerivative[(y * w + x) * 2 + 0] = movingCorrelationDerivative * mappedMovingPixelGradient_x;
    d_FixedDerivative[(y * w + x) * 2 + 1] = movingCorrelationDerivative * mappedMovingPixelGradient_y;
    d_MovingDerivative[(y * w + x) * 2 + 0] = fixedCorrelationDerivative * mappedFixedPixelGradient_x;
    d_MovingDerivative[(y * w + x) * 2 + 1] = fixedCorrelationDerivative * mappedFixedPixelGradient_y;

    // Kept in the native signature for API compatibility with existing wheels.
    (void)robustThreshold;

//     if(x == 32 && y == 64){
//         printf("d_ValueResult:%f, d_FixedDerivative:[%f,%f], d_MovingDerivative:[%f,%f]\n",
//             d_ValueResult[y * w + x], d_FixedDerivative[(y * w + x) * 2 + 0], d_FixedDerivative[(y * w + x) * 2 + 1],
//                                       d_MovingDerivative[(y * w + x) * 2 + 0], d_MovingDerivative[(y * w + x) * 2 + 1]);
//     }
}

__global__ void cudaStoreRowValueKernal(int w, int h, double *d_ValueResult, double *t_Value){

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= h){return;}
    t_Value[index] = 0;
    int trueIndex;
    for(int i = 0; i < w; i++){
        trueIndex = index * w + i;
        if(trueIndex >= w * h){break;}
        t_Value[index] += d_ValueResult[trueIndex];
    }
}

__global__ void cudaStoreColValueKernal(int w, int h, double *t_Value, int currentIteration, int minIteration,
                                        double *d_EnergyValues, double *d_TotalEnergy){
    //
    double d_Value = 0;

    for(int i = 0; i < h; i++){
        d_Value += t_Value[i];
    }

    d_Value /= (w * h);

//     printf("ConvergenceValue:%f validPoint:%d\n",d_Value,w*h);

    d_EnergyValues[currentIteration % minIteration] = d_Value;
    *d_TotalEnergy += d_Value;
}

__global__ void cudaSmoothByRowKernal(int w, int h, CudaFieldValueType *d_Derivative, CudaFieldValueType *t_Derivative){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    int _1x = x - 1; if(_1x < 0){ _1x = 0; }    int x1_ = x + 1; if(x1_ > w - 1){ x1_ = w - 1; }
    int _2x = x - 2; if(_2x < 0){ _2x = 0; }    int x2_ = x + 2; if(x2_ > w - 1){ x2_ = w - 1; }
    int _3x = x - 3; if(_3x < 0){ _3x = 0; }    int x3_ = x + 3; if(x3_ > w - 1){ x3_ = w - 1; }
    int _4x = x - 4; if(_4x < 0){ _4x = 0; }    int x4_ = x + 4; if(x4_ > w - 1){ x4_ = w - 1; }
    int _5x = x - 5; if(_5x < 0){ _5x = 0; }    int x5_ = x + 5; if(x5_ > w - 1){ x5_ = w - 1; }
    int _6x = x - 6; if(_6x < 0){ _6x = 0; }    int x6_ = x + 6; if(x6_ > w - 1){ x6_ = w - 1; }

    t_Derivative[(y * w + x) * 2 + 0] = 0.00108015 * d_Derivative[(y * w + _6x) * 2 + 0] +
                                        0.00454336 * d_Derivative[(y * w + _5x) * 2 + 0] +
                                        0.0162247  * d_Derivative[(y * w + _4x) * 2 + 0] +
                                        0.0478092  * d_Derivative[(y * w + _3x) * 2 + 0] +
                                        0.111843   * d_Derivative[(y * w + _2x) * 2 + 0] +
                                        0.196933   * d_Derivative[(y * w + _1x) * 2 + 0] +
                                        0.243132   * d_Derivative[(y * w +  x ) * 2 + 0] +
                                        0.196933   * d_Derivative[(y * w + x1_) * 2 + 0] +
                                        0.111843   * d_Derivative[(y * w + x2_) * 2 + 0] +
                                        0.0478092  * d_Derivative[(y * w + x3_) * 2 + 0] +
                                        0.0162247  * d_Derivative[(y * w + x4_) * 2 + 0] +
                                        0.00454336 * d_Derivative[(y * w + x5_) * 2 + 0] +
                                        0.00108015 * d_Derivative[(y * w + x6_) * 2 + 0];

    t_Derivative[(y * w + x) * 2 + 1] = 0.00108015 * d_Derivative[(y * w + _6x) * 2 + 1] +
                                        0.00454336 * d_Derivative[(y * w + _5x) * 2 + 1] +
                                        0.0162247  * d_Derivative[(y * w + _4x) * 2 + 1] +
                                        0.0478092  * d_Derivative[(y * w + _3x) * 2 + 1] +
                                        0.111843   * d_Derivative[(y * w + _2x) * 2 + 1] +
                                        0.196933   * d_Derivative[(y * w + _1x) * 2 + 1] +
                                        0.243132   * d_Derivative[(y * w +  x ) * 2 + 1] +
                                        0.196933   * d_Derivative[(y * w + x1_) * 2 + 1] +
                                        0.111843   * d_Derivative[(y * w + x2_) * 2 + 1] +
                                        0.0478092  * d_Derivative[(y * w + x3_) * 2 + 1] +
                                        0.0162247  * d_Derivative[(y * w + x4_) * 2 + 1] +
                                        0.00454336 * d_Derivative[(y * w + x5_) * 2 + 1] +
                                        0.00108015 * d_Derivative[(y * w + x6_) * 2 + 1];

//     if(x == 32 && y == 64){
//         // for(int i = -6; i <= 6; i++){
//         //     printf("row (%d)d_Derivative:[%f,%f]\n",i,
//         //         d_Derivative[(y * w + x + i) * 2 + 0],d_Derivative[(y * w + x + i) * 2 + 1]);
//         // }
//         printf("row d_Derivative:[%f,%f] t_Derivative:[%f,%f]\n",
//             d_Derivative[(y * w + x) * 2 + 0],d_Derivative[(y * w + x) * 2 + 1],
//             t_Derivative[(y * w + x) * 2 + 0],t_Derivative[(y * w + x) * 2 + 1]);
//     }
}

__global__ void cudaSmoothByColKernal(int w, int h, CudaFieldValueType *d_Derivative, CudaFieldValueType *t_Derivative){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    int _1y = y - 1; if(_1y < 0){ _1y = 0; }    int y1_ = y + 1; if(y1_ > h - 1){ y1_ = h - 1; }
    int _2y = y - 2; if(_2y < 0){ _2y = 0; }    int y2_ = y + 2; if(y2_ > h - 1){ y2_ = h - 1; }
    int _3y = y - 3; if(_3y < 0){ _3y = 0; }    int y3_ = y + 3; if(y3_ > h - 1){ y3_ = h - 1; }
    int _4y = y - 4; if(_4y < 0){ _4y = 0; }    int y4_ = y + 4; if(y4_ > h - 1){ y4_ = h - 1; }
    int _5y = y - 5; if(_5y < 0){ _5y = 0; }    int y5_ = y + 5; if(y5_ > h - 1){ y5_ = h - 1; }
    int _6y = y - 6; if(_6y < 0){ _6y = 0; }    int y6_ = y + 6; if(y6_ > h - 1){ y6_ = h - 1; }

    t_Derivative[(y * w + x) * 2 + 0] = 0.00108015 * d_Derivative[(_6y * w + x) * 2 + 0] +
                                        0.00454336 * d_Derivative[(_5y * w + x) * 2 + 0] +
                                        0.0162247  * d_Derivative[(_4y * w + x) * 2 + 0] +
                                        0.0478092  * d_Derivative[(_3y * w + x) * 2 + 0] +
                                        0.111843   * d_Derivative[(_2y * w + x) * 2 + 0] +
                                        0.196933   * d_Derivative[(_1y * w + x) * 2 + 0] +
                                        0.243132   * d_Derivative[( y  * w + x) * 2 + 0] +
                                        0.196933   * d_Derivative[(y1_ * w + x) * 2 + 0] +
                                        0.111843   * d_Derivative[(y2_ * w + x) * 2 + 0] +
                                        0.0478092  * d_Derivative[(y3_ * w + x) * 2 + 0] +
                                        0.0162247  * d_Derivative[(y4_ * w + x) * 2 + 0] +
                                        0.00454336 * d_Derivative[(y5_ * w + x) * 2 + 0] +
                                        0.00108015 * d_Derivative[(y6_ * w + x) * 2 + 0];

    t_Derivative[(y * w + x) * 2 + 1] = 0.00108015 * d_Derivative[(_6y * w + x) * 2 + 1] +
                                        0.00454336 * d_Derivative[(_5y * w + x) * 2 + 1] +
                                        0.0162247  * d_Derivative[(_4y * w + x) * 2 + 1] +
                                        0.0478092  * d_Derivative[(_3y * w + x) * 2 + 1] +
                                        0.111843   * d_Derivative[(_2y * w + x) * 2 + 1] +
                                        0.196933   * d_Derivative[(_1y * w + x) * 2 + 1] +
                                        0.243132   * d_Derivative[( y  * w + x) * 2 + 1] +
                                        0.196933   * d_Derivative[(y1_ * w + x) * 2 + 1] +
                                        0.111843   * d_Derivative[(y2_ * w + x) * 2 + 1] +
                                        0.0478092  * d_Derivative[(y3_ * w + x) * 2 + 1] +
                                        0.0162247  * d_Derivative[(y4_ * w + x) * 2 + 1] +
                                        0.00454336 * d_Derivative[(y5_ * w + x) * 2 + 1] +
                                        0.00108015 * d_Derivative[(y6_ * w + x) * 2 + 1];
//     if(x == 32 && y == 64){
//         // for(int i = -6; i <= 6; i++){
//         //     printf("col (%d)d_Derivative:[%f,%f]\n",i,
//         //         d_Derivative[((y + i) * w + x) * 2 + 0],d_Derivative[((y + i) * w + x) * 2 + 1]);
//         // }
//         printf("col d_Derivative:[%f,%f] t_Derivative:[%f,%f]\n",
//             d_Derivative[(y * w + x) * 2 + 0],d_Derivative[(y * w + x) * 2 + 1],
//             t_Derivative[(y * w + x) * 2 + 0],t_Derivative[(y * w + x) * 2 + 1]);
//     }
}

__global__ void cudaCheckBoundaryKernal(int w, int h, CudaFieldValueType *d_Derivative){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    if(x == 0 || y == 0 || x == w - 1 || y == h - 1){
        d_Derivative[(y * w + x) * 2 + 0] = 0;
        d_Derivative[(y * w + x) * 2 + 1] = 0;
    }
}

__global__ void cudaGetRowNormKernal(int w, int h, int spacing, CudaFieldValueType *d_Derivative, double *t_Scale){
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if(y >= h){return;}
    double maxNorm = 0;
    for(int x = 0; x < w; x++){
        double localNorm = sqrt((d_Derivative[(y * w + x) * 2 + 0] / spacing) * (d_Derivative[(y * w + x) * 2 + 0] / spacing) +
                                (d_Derivative[(y * w + x) * 2 + 1] / spacing) * (d_Derivative[(y * w + x) * 2 + 1] / spacing));
        if(localNorm > maxNorm){
            maxNorm = localNorm;
        }
    }
    t_Scale[y] = maxNorm;
}

__global__ void cudaGetColNormKernal(int w, int h, double *t_Scale, double *d_Scale, double maxStep){
    *d_Scale = 0;
    for(int y = 0; y < h; y++){
        if(t_Scale[y] > *d_Scale){
            *d_Scale = t_Scale[y];
        }
    }
    if(*d_Scale > 0){
        *d_Scale = maxStep / (*d_Scale);
    }
    else{
        *d_Scale = 0;
    }
}

__global__ void cudaScaleFieldKernal(int w, int h, CudaFieldValueType *d_Derivative, double *d_Scale){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    // if(x == 32 && y == 64){
    //     printf("input:[%f,%f] d_Scale:%f ",d_Derivative[(y * w + x) * 2 + 0],d_Derivative[(y * w + x) * 2 + 1],*d_Scale);
    // }

    d_Derivative[(y * w + x) * 2 + 0] *= (*d_Scale);
    d_Derivative[(y * w + x) * 2 + 1] *= (*d_Scale);

    // if(x == 32 && y == 64){
    //     printf("output:[%f,%f]\n",d_Derivative[(y * w + x) * 2 + 0],d_Derivative[(y * w + x) * 2 + 1]);
    // }
}

__global__ void cudaComposeFieldKernal(int w, int h, int spacing, CudaFieldValueType *d_DisplacementField, CudaFieldValueType *d_WarpingField,
                                       double offset_x, double offset_y, CudaFieldValueType *t_OutputField){

    //
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    double pointIn1_x = x * spacing + offset_x;
    double pointIn1_y = y * spacing + offset_y;

    double warpVector_x = d_WarpingField[(y * w + x) * 2 + 0];
    double warpVector_y = d_WarpingField[(y * w + x) * 2 + 1];

    double pointIn2_x = pointIn1_x + warpVector_x;
    double pointIn2_y = pointIn1_y + warpVector_y;

    double continuousIndex_x = (pointIn2_x - offset_x) / spacing;
    double continuousIndex_y = (pointIn2_y - offset_y) / spacing;

    double displacement_x = 0;
    double displacement_y = 0;

    if(continuousIndex_x < -0.5 || continuousIndex_x >= w - 0.5 ||
       continuousIndex_y < -0.5 || continuousIndex_y >= h - 0.5){
    }else{
        int baseIndex_x = (int)continuousIndex_x;
        int baseIndex_y = (int)continuousIndex_y;

        double distance_x = continuousIndex_x - baseIndex_x;
        double distance_y = continuousIndex_y - baseIndex_y;

        double totalOverlap = 0;

        for(unsigned int counter = 0; counter < 4; ++counter){

            double overlap = 1.0;
            unsigned int upper = counter;

            int neighIndex_x;
            int neighIndex_y;

            if(upper & 1){
                neighIndex_x = baseIndex_x + 1;
                if(neighIndex_x > w - 1){
                    neighIndex_x = w - 1;
                }
                overlap *= distance_x;
            }
            else{
                neighIndex_x = baseIndex_x;
                if(neighIndex_x < 0){
                    neighIndex_x = 0;
                }
                overlap *= 1.0 - distance_x;
            }
            upper >>= 1;
            if (upper & 1){
                neighIndex_y = baseIndex_y + 1;
                if(neighIndex_y > h - 1){
                    neighIndex_y = h - 1;
                }
                overlap *= distance_y;
            }
            else{
                neighIndex_y = baseIndex_y;
                if(baseIndex_y < 0){
                    baseIndex_y = 0;
                }
                overlap *= 1.0 - distance_y;
            }

            if(overlap){
                double input_x = d_DisplacementField[(neighIndex_y * w + neighIndex_x) * 2 + 0];
                double input_y = d_DisplacementField[(neighIndex_y * w + neighIndex_x) * 2 + 1];
                displacement_x += overlap * input_x;
                displacement_y += overlap * input_y;
                totalOverlap += overlap;
            }

            if(totalOverlap == 1.0){
                break;
            }
        }
    }

    double pointIn3_x = pointIn2_x + displacement_x;
    double pointIn3_y = pointIn2_y + displacement_y;

    t_OutputField[(y * w + x) * 2 + 0] = pointIn3_x - pointIn1_x;
    t_OutputField[(y * w + x) * 2 + 1] = pointIn3_y - pointIn1_y;

//     if(x == 32 && y == 64){
//         printf(" cuda warpVector:[%f,%f] displacement:[%f,%f] outDisplacement:[%f,%f]\n",
//             warpVector_x,warpVector_y,displacement_x,displacement_y,pointIn3_x-pointIn1_x,pointIn3_y-pointIn1_y);
//     }
}

__global__ void cudaCalcScaledNormImageKernal(int w, int h, int spacing, CudaFieldValueType *t_ComposeField, CudaFieldValueType *t_ScaledNormImage){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    t_ScaledNormImage[y * w + x] = sqrt((t_ComposeField[(y * w + x) * 2 + 0] / spacing) *
                                        (t_ComposeField[(y * w + x) * 2 + 0] / spacing) +
                                        (t_ComposeField[(y * w + x) * 2 + 1] / spacing) *
                                        (t_ComposeField[(y * w + x) * 2 + 1] / spacing));
    t_ComposeField[(y * w + x) * 2 + 0] = -t_ComposeField[(y * w + x) * 2 + 0];
    t_ComposeField[(y * w + x) * 2 + 1] = -t_ComposeField[(y * w + x) * 2 + 1];
}

__global__ void cudaGetRowMeanMaxErrorNormKernal(int w, int h, CudaFieldValueType *t_ScaledNormImage, double *t_MeanErrorNorm,
                                                 double *t_MaxErrorNorm){
    //
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= h){return;}
    t_MeanErrorNorm[index] = 0;
    t_MaxErrorNorm[index] = 0;
    int trueIndex;
    for(int i = 0; i < w; i++){
        trueIndex = index * w + i;
        if(trueIndex >= w * h){break;}
        t_MeanErrorNorm[index] += t_ScaledNormImage[trueIndex];
        if(t_MaxErrorNorm[index] < t_ScaledNormImage[trueIndex]){
            t_MaxErrorNorm[index] = t_ScaledNormImage[trueIndex];
        }
    }
}

__global__ void cudaGetColMeanMaxErrorNormKernal(int w, int h, double *t_MeanErrorNorm, double *t_MaxErrorNorm,
                                                 double *d_MeanErrorNorm, double *d_MaxErrorNorm){
    //
    *d_MeanErrorNorm = 0;
    *d_MaxErrorNorm = 0;
    for(int y = 0; y < h; y++){
        *d_MeanErrorNorm += t_MeanErrorNorm[y];
        if(t_MaxErrorNorm[y] > *d_MaxErrorNorm){
            *d_MaxErrorNorm = t_MaxErrorNorm[y];
        }
    }
    *d_MeanErrorNorm /= (w * h);
}

__global__ void cudaInverseFieldKernal(int w, int h, CudaFieldValueType *t_ComposeField, CudaFieldValueType *t_ScaledNormImage, double epsilon,
                                       double *d_MaxErrorNorm, CudaFieldValueType *d_OutputField){
    //
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    double update_x = t_ComposeField[(y * w + x) * 2 + 0];
    double update_y = t_ComposeField[(y * w + x) * 2 + 1];

    double scaledNorm = t_ScaledNormImage[y * w + x];

//     if(x == 32 && y == 64){
//         printf("cuda original update:[%f,%f] scaledNorm:%f ",update_x,update_y,scaledNorm);
//     }

    if(scaledNorm > epsilon * (*d_MaxErrorNorm)){
        update_x *= (epsilon * (*d_MaxErrorNorm) / scaledNorm);
        update_y *= (epsilon * (*d_MaxErrorNorm) / scaledNorm);
    }

    update_x = d_OutputField[(y * w + x) * 2 + 0] + update_x * epsilon;
    update_y = d_OutputField[(y * w + x) * 2 + 1] + update_y * epsilon;

//     if(x == 32 && y == 64){
//         printf("cuda final update:[%f,%f]\n",update_x,update_y);
//     }

    if(x == 0 || x == w - 1 || y == 0 || y == h - 1){
        d_OutputField[(y * w + x) * 2 + 0] = 0;
        d_OutputField[(y * w + x) * 2 + 1] = 0;
    }
    else{
        d_OutputField[(y * w + x) * 2 + 0] = update_x;
        d_OutputField[(y * w + x) * 2 + 1] = update_y;
    }
}

/*-------------------------------------------------------------------------------------------------------------*/

__global__ void cudaCopyDisplacementFieldKernal(int w, int h, CudaFieldValueType *d_DisplacementField, CudaFieldValueType *o_DisplacementField,
                                                double maxDisplacement){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}
    int index = y * w + x;
    double displacement_x = o_DisplacementField[index * 2 + 0];
    double displacement_y = o_DisplacementField[index * 2 + 1];

    // Keep SyN as a low-frequency mechanical correction, not a defect-erasing free warp.
    if(maxDisplacement > 0){
        displacement_x = ClampValue(displacement_x, -maxDisplacement, maxDisplacement);
        displacement_y = ClampValue(displacement_y, -maxDisplacement, maxDisplacement);
    }

    d_DisplacementField[index * 2 + 0] = displacement_x;
    d_DisplacementField[index * 2 + 1] = displacement_y;
}

/*-------------------------------------------------------------------------------------------------------------*/

__global__ void cudaInitDisplacementFieldKernal(int w, int h, CudaFieldValueType *d_DisplacementField){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}
    int index = y * w + x;
    d_DisplacementField[index * 2 + 0] = 0;
    d_DisplacementField[index * 2 + 1] = 0;
}

__global__ void cudaAdaptDisplacementFieldKernal(int w, int h, int spacingOld, int spacingNew, CudaFieldValueType *d_OldField,
                                                 double offsetOld_x, double offsetOld_y, double offsetNew_x, double offsetNew_y,
                                                 CudaFieldValueType *d_NewField){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outW = w / spacingNew;
    int outH = h / spacingNew;
    if(x >= outW || y >= outH){return;}

    double outputPoint_x = x * spacingNew + offsetNew_x;
    double outputPoint_y = y * spacingNew + offsetNew_y;

    double inputIndex_x = (outputPoint_x - offsetOld_x) / spacingOld;
    double inputIndex_y = (outputPoint_y - offsetOld_y) / spacingOld;

    double displacement_x = 0;
    double displacement_y = 0;

    /*-------------------------------------------------------------------------------------------------------------*/

    int basei[2];

    basei[0] = (int)inputIndex_x;if(basei[0] < 0){basei[0] = 0;}
    double distance0 = inputIndex_x - basei[0];

    basei[1] = (int)inputIndex_y;if(basei[1] < 0){basei[1] = 0;}
    double distance1 = inputIndex_y - basei[1];

    double val00_x = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 0];
    double val00_y = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 1];

    if (distance0 <= 0. && distance1 <= 0.){
        displacement_x = val00_x;
        displacement_y = val00_y;
    }
    else if (distance1 <= 0.){
        ++basei[0];
        if(basei[0] >= w / spacingOld){
            displacement_x = val00_x;
            displacement_y = val00_y;
        }
        else{
            double val10_x = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 0];
            double val10_y = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 1];
            displacement_x = val00_x + (val10_x - val00_x) * distance0;
            displacement_y = val00_y + (val10_y - val00_y) * distance0;
        }
    }
    else if (distance0 <= 0.){
        ++basei[1];
        if(basei[1] >= h / spacingOld){
            displacement_x = val00_x;
            displacement_y = val00_y;
        }
        else{
            double val01_x = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 0];
            double val01_y = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 1];
            displacement_x = val00_x + (val01_x - val00_x) * distance1;
            displacement_y = val00_y + (val01_y - val00_y) * distance1;
        }
    }
    else{
        ++basei[0];
        if(basei[0] >= w / spacingOld){
            --basei[0];
            ++basei[1];
            if(basei[1] >= h / spacingOld){
                displacement_x = val00_x;
                displacement_y = val00_y;
            }
            else{
                double val01_x = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 0];
                double val01_y = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 1];
                displacement_x = val00_x + (val01_x - val00_x) * distance1;
                displacement_y = val00_y + (val01_y - val00_y) * distance1;
            }
        }
        else{
            double val10_x = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 0];
            double val10_y = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 1];

            double valx0_x = val00_x + (val10_x - val00_x) * distance0;
            double valx0_y = val00_y + (val10_y - val00_y) * distance0;

            ++basei[1];
            if(basei[1] >= h / spacingOld){
                displacement_x = valx0_x;
                displacement_y = valx0_y;
            }
            else{
                double val11_x = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 0];
                double val11_y = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 1];
                --basei[0];
                double val01_x = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 0];
                double val01_y = d_OldField[(basei[1] * (w / spacingOld) + basei[0]) * 2 + 1];
                double valx1_x = val01_x + (val11_x - val01_x) * distance0;
                double valx1_y = val01_y + (val11_y - val01_y) * distance0;
                displacement_x = valx0_x + (valx1_x - valx0_x) * distance1;
                displacement_y = valx0_y + (valx1_y - valx0_y) * distance1;
            }
        }
    }

    /*-------------------------------------------------------------------------------------------------------------*/

    // if(inputIndex_x < -0.5 || inputIndex_x >= (w / spacingOld) - 0.5 ||
    //    inputIndex_y < -0.5 || inputIndex_y >= (h / spacingOld) - 0.5){
    // }else{
    //     int baseIndex_x = (int)inputIndex_x;
    //     int baseIndex_y = (int)inputIndex_y;

    //     double distance_x = inputIndex_x - baseIndex_x;
    //     double distance_y = inputIndex_y - baseIndex_y;

    //     double totalOverlap = 0;

    //     for(unsigned int counter = 0; counter < 4; ++counter){

    //         double overlap = 1.0;
    //         unsigned int upper = counter;

    //         int neighIndex_x;
    //         int neighIndex_y;

    //         if(upper & 1){
    //             neighIndex_x = baseIndex_x + 1;
    //             if(neighIndex_x > (w / spacingOld) - 1){
    //                 neighIndex_x = (w / spacingOld) - 1;
    //             }
    //             overlap *= distance_x;
    //         }
    //         else{
    //             neighIndex_x = baseIndex_x;
    //             if(neighIndex_x < 0){
    //                 neighIndex_x = 0;
    //             }
    //             overlap *= 1.0 - distance_x;
    //         }
    //         upper >>= 1;
    //         if (upper & 1){
    //             neighIndex_y = baseIndex_y + 1;
    //             if(neighIndex_y > (h / spacingOld) - 1){
    //                 neighIndex_y = (h / spacingOld) - 1;
    //             }
    //             overlap *= distance_y;
    //         }
    //         else{
    //             neighIndex_y = baseIndex_y;
    //             if(baseIndex_y < 0){
    //                 baseIndex_y = 0;
    //             }
    //             overlap *= 1.0 - distance_y;
    //         }

    //         if(overlap){
    //             double input_x = d_OldField[(neighIndex_y * (w / spacingOld) + neighIndex_x) * 2 + 0];
    //             double input_y = d_OldField[(neighIndex_y * (w / spacingOld) + neighIndex_x) * 2 + 1];
    //             displacement_x += overlap * input_x;
    //             displacement_y += overlap * input_y;
    //             totalOverlap += overlap;
    //         }

    //         if(totalOverlap == 1.0){
    //             break;
    //         }
    //     }
    // }

    /*-------------------------------------------------------------------------------------------------------------*/

//     if(x == 32 && y == 64){
//         printf("outIndex:[%d,%d] inIndex:[%f,%f] value:[%f,%f]\n",x,y,inputIndex_x,inputIndex_y,displacement_x,displacement_y);
//     }

    d_NewField[(y * outW + x) * 2 + 0] = displacement_x;
    d_NewField[(y * outW + x) * 2 + 1] = displacement_y;
}

__global__ void cudaApplyTransformKernal(int w, int h, float *d_OrigImage, double *d_InitialTransform,
                                         double *d_AffineTransform, CudaFieldValueType *d_OutputFieldFinal,
                                         CudaFieldValueType *d_OutputInverseFieldFinal, int outputDirection,
                                         int *d_ResultImage, float *d_TransformMap){
    //
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= w || y >= h){return;}

    double true_x = 0;
    double true_y = 0;

    if(outputDirection == 1){
        if(!MapFixedToMovingSource(w, h, x, y, d_InitialTransform, d_AffineTransform,
                                   d_OutputInverseFieldFinal, true_x, true_y)){
            d_ResultImage[y * w + x] = 0;
            if(d_TransformMap){
                d_TransformMap[(y * w + x) * 2 + 0] = -1.0f;
                d_TransformMap[(y * w + x) * 2 + 1] = -1.0f;
            }
            return;
        }
    }
    else{
        double temp_x = x + d_OutputFieldFinal[(y * w + x) * 2 + 0];
        double temp_y = y + d_OutputFieldFinal[(y * w + x) * 2 + 1];

        double temp_x_2 = d_AffineTransform[4] + d_AffineTransform[0] * temp_x
                                               + d_AffineTransform[1] * temp_y;
        double temp_y_2 = d_AffineTransform[5] + d_AffineTransform[2] * temp_x
                                               + d_AffineTransform[3] * temp_y;

        true_x = d_InitialTransform[4] + d_InitialTransform[0] * temp_x_2
                                       + d_InitialTransform[1] * temp_y_2;
        true_y = d_InitialTransform[5] + d_InitialTransform[2] * temp_x_2
                                       + d_InitialTransform[3] * temp_y_2;
    }

    if(true_x < -0.5 || true_x > w - 0.5 || true_y < -0.5 || true_y > h - 0.5){
        d_ResultImage[y * w + x] = 0;
        if(d_TransformMap){
            d_TransformMap[(y * w + x) * 2 + 0] = -1.0f;
            d_TransformMap[(y * w + x) * 2 + 1] = -1.0f;
        }
    }
    else if(outputDirection == 1){
        if(d_TransformMap){
            d_TransformMap[(y * w + x) * 2 + 0] = (float)true_x;
            d_TransformMap[(y * w + x) * 2 + 1] = (float)true_y;
        }
        // A target pixel can cover more than one template pixel where the
        // deformation compresses printed content.  Integrating that footprint
        // avoids the aliased gray/black halos produced by center-only bilinear
        // sampling while leaving ordinary sub-pixel interpolation untouched.
        double sourceXp, sourceYp, sourceXm, sourceYm;
        double sourceXyp, sourceYyp, sourceXym, sourceYym;
        bool validX = MapFixedToMovingSource(w, h, x + 0.5, y, d_InitialTransform, d_AffineTransform,
                                             d_OutputInverseFieldFinal, sourceXp, sourceYp)
                   && MapFixedToMovingSource(w, h, x - 0.5, y, d_InitialTransform, d_AffineTransform,
                                             d_OutputInverseFieldFinal, sourceXm, sourceYm);
        bool validY = MapFixedToMovingSource(w, h, x, y + 0.5, d_InitialTransform, d_AffineTransform,
                                             d_OutputInverseFieldFinal, sourceXyp, sourceYyp)
                   && MapFixedToMovingSource(w, h, x, y - 0.5, d_InitialTransform, d_AffineTransform,
                                             d_OutputInverseFieldFinal, sourceXym, sourceYym);

        double dxX = validX ? sourceXp - sourceXm : 0.0;
        double dxY = validX ? sourceYp - sourceYm : 0.0;
        double dyX = validY ? sourceXyp - sourceXym : 0.0;
        double dyY = validY ? sourceYyp - sourceYym : 0.0;
        double xFootprint = sqrt(dxX * dxX + dxY * dxY);
        double yFootprint = sqrt(dyX * dyX + dyY * dyY);
        constexpr double compressionThreshold = 1.05;

        double outputValue;
        if(validX && validY && xFootprint > compressionThreshold && yFootprint > compressionThreshold){
            outputValue = 0.25 * (
                EvaluateValueClamped(w, h, d_OrigImage, true_x - 0.25 * dxX - 0.25 * dyX,
                                                             true_y - 0.25 * dxY - 0.25 * dyY) +
                EvaluateValueClamped(w, h, d_OrigImage, true_x + 0.25 * dxX - 0.25 * dyX,
                                                             true_y + 0.25 * dxY - 0.25 * dyY) +
                EvaluateValueClamped(w, h, d_OrigImage, true_x - 0.25 * dxX + 0.25 * dyX,
                                                             true_y - 0.25 * dxY + 0.25 * dyY) +
                EvaluateValueClamped(w, h, d_OrigImage, true_x + 0.25 * dxX + 0.25 * dyX,
                                                             true_y + 0.25 * dxY + 0.25 * dyY));
        }
        else if(validX && xFootprint > compressionThreshold){
            outputValue = 0.5 * (
                EvaluateValueClamped(w, h, d_OrigImage, true_x - 0.25 * dxX, true_y - 0.25 * dxY) +
                EvaluateValueClamped(w, h, d_OrigImage, true_x + 0.25 * dxX, true_y + 0.25 * dxY));
        }
        else if(validY && yFootprint > compressionThreshold){
            outputValue = 0.5 * (
                EvaluateValueClamped(w, h, d_OrigImage, true_x - 0.25 * dyX, true_y - 0.25 * dyY) +
                EvaluateValueClamped(w, h, d_OrigImage, true_x + 0.25 * dyX, true_y + 0.25 * dyY));
        }
        else{
            outputValue = EvaluateValueClamped(w, h, d_OrigImage, true_x, true_y);
        }
        d_ResultImage[y * w + x] = __double2int_rn(outputValue);
    }
    else{
        if(d_TransformMap){
            d_TransformMap[(y * w + x) * 2 + 0] = (float)true_x;
            d_TransformMap[(y * w + x) * 2 + 1] = (float)true_y;
        }
        d_ResultImage[y * w + x] = __double2int_rn(EvaluateValue(w, h, d_OrigImage, true_x, true_y));
    }
//     if(x == 32 && y == 64){
//         printf("d_ResultImage[%d,%d]->[%f,%f]: %d\n",x,y,true_x,true_y,d_ResultImage[y * w + x]);
//     }
}

/*-------------------------------------------------------------------------------------------------------------*/

void EXEC_cudaEstimateScales(void *d_InitialTransform, void *d_AffineTransform, double offset_x,
                             double offset_y, double maxPixel_x, double maxPixel_y, void *d_Scales){

    cudaEstimateScalesKermal<<<1,6>>>((double*)d_InitialTransform, (double*)d_AffineTransform,
                                      offset_x, offset_y, maxPixel_x, maxPixel_y, (double*)d_Scales);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaEstimateLearningRate(void *d_InitialTransform, void *d_AffineTransform, void *d_Gradient, double offset_x,
                                   double offset_y, double maxPixel_x, double maxPixel_y, void *d_LearningRate){

    cudaEstimateLearningRateKernal<<<1,1>>>((double*)d_InitialTransform, (double*)d_AffineTransform, (double*)d_Gradient,
                                            offset_x, offset_y, maxPixel_x, maxPixel_y, (double*)d_LearningRate);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaCalculateAffine(int w, int h, int spacing, void *d_FixedImage, void *d_MovingImage, void *d_InitialTransform,
                              void *d_AffineTransform, void *d_VirtualPointSet, bool ComputeDerivative, int NumberOfDomainPoints,
                              int pointsInWidth, int pointsInHeightCeil, void *d_DerivativeResult, void *d_ValueResult){

    dim3 block(256);
    dim3 grid((NumberOfDomainPoints + block.x - 1) / block.x);

    cudaCalculateAffineKernal<<<grid,block>>>(w, h, spacing, (float*)d_FixedImage, (float*)d_MovingImage,
        (double*)d_InitialTransform, (double*)d_AffineTransform, (double*)d_VirtualPointSet, ComputeDerivative,
        NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil, (CudaAffineMetricValueType*)d_DerivativeResult,
        (CudaAffineMetricValueType*)d_ValueResult);
    CUDA_CHECK(cudaGetLastError());

    // cudaDeviceSynchronize();
    // printf("EXEC_cudaCalculateAffine after kernal function : %s\n",cudaGetErrorString(cudaGetLastError()));
}

void EXEC_cudaProcessGradient(int w, int h, int spacing, void *d_DerivativeResult, void *d_ValueResult, void *d_Scales,
                              int NumberOfDomainPoints, int pointsInWidth, int pointsInHeightCeil, int iteration,
                              int minIteration, void *d_Gradient, void *d_Value, void *d_EnergyValues, void *d_TotalEnergy,
                              void *t_Gradient, void *t_Value, void *t_ValidPointCount){

    dim3 block(256);
    dim3 grid(pointsInHeightCeil);

    cudaGetRowGradientKernal<<<grid,block>>>(w, h, spacing, (CudaAffineMetricValueType*)d_DerivativeResult,
                                             (CudaAffineMetricValueType*)d_ValueResult,
                                             NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil, (double*)t_Gradient,
                                             (CudaAffineMetricValueType*)t_Value,
                                             (CudaAffineMetricValueType*)t_ValidPointCount);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
//     printf("cudaGetRowGradientKernal after kernal function : %s\n",cudaGetErrorString(cudaGetLastError()));

    cudaGetColGradientKernal<<<1,1>>>(w, h, spacing, (double*)t_Gradient, (CudaAffineMetricValueType*)t_Value,
        (CudaAffineMetricValueType*)t_ValidPointCount,
        (double*)d_Scales, NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil, iteration, minIteration,
        (double*)d_Gradient, (double*)d_Value, (double*)d_EnergyValues, (double*)d_TotalEnergy);
    CUDA_CHECK(cudaGetLastError());

    cudaDeviceSynchronize();
//     printf("cudaGetColGradientKernal after kernal function : %s\n",cudaGetErrorString(cudaGetLastError()));
}

void EXEC_cudaProcessValue(int w, int h, int spacing, void *d_ValueResult, int NumberOfDomainPoints, int pointsInWidth,
                           int pointsInHeightCeil, void *d_Value, void *t_Value, void *t_ValidPointCount){

    dim3 block(256);
    dim3 grid(pointsInHeightCeil);

    cudaGetRowValueKernal<<<grid,block>>>(w, h, spacing, (CudaAffineMetricValueType*)d_ValueResult, NumberOfDomainPoints, pointsInWidth,
                                          pointsInHeightCeil, (CudaAffineMetricValueType*)t_Value,
                                          (CudaAffineMetricValueType*)t_ValidPointCount);
    CUDA_CHECK(cudaGetLastError());

    cudaGetColValueKernal<<<1,1>>>(w, h, spacing, (CudaAffineMetricValueType*)t_Value,
                                   (CudaAffineMetricValueType*)t_ValidPointCount, NumberOfDomainPoints,
                                   pointsInWidth, pointsInHeightCeil, (double*)d_Value);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaResetGradient(void *d_Gradient, void *d_LearningRate){

    cudaResetGradientKernal<<<1,6>>>((double*)d_Gradient, (double*)d_LearningRate);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaResetTransform(void *d_AffineTransform, void *d_Gradient){

    cudaResetTransformKernal<<<1,1>>>((double*)d_AffineTransform, (double*)d_Gradient);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaModifyGradient(void *d_Gradient, void *g_LearningRate){

    cudaModifyGradientKernal<<<1,6>>>((double*)d_Gradient, (double*)g_LearningRate);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaUpdateTransform(void *d_AffineTransform, void *d_Gradient){

    cudaUpdateTransformKernal<<<1,1>>>((double*)d_AffineTransform, (double*)d_Gradient);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaGetConvergenceValue(void *d_EnergyValues, void *d_TotalEnergy, int iteration, int minIteration,
                                  void *d_ConvergenceValue, void *d_DeltaLattice, void *d_OmegaLattice){

    cudaGetLatticeKernal<<<1,minIteration>>>((double*)d_EnergyValues, (double*)d_TotalEnergy, iteration, minIteration,
                                             (double*)d_DeltaLattice, (double*)d_OmegaLattice);
    CUDA_CHECK(cudaGetLastError());
    cudaGetConvergenceValueKernal<<<1,1>>>((double*)d_DeltaLattice, (double*)d_OmegaLattice, (double*)d_ConvergenceValue);
    CUDA_CHECK(cudaGetLastError());

}

void EXEC_cudaSetGoldenSection(void *d_LearningRate, void *d_Gradient, void *a, void *b, void *c){
    cudaSetGoldenSectionKernal<<<1,1>>>((double*)d_LearningRate, (double*)d_Gradient,
                                       (double*)a, (double*)b, (double*)c);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaFindGoldenSection(void *a, void *b, void *c, void *x, void *metricb, void *metricx){
    cudaFindGoldenSectionKernal<<<1,1>>>((double*)a, (double*)b, (double*)c, (double*)x, (double*)metricb, (double*)metricx);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaCalcGoldenSection(int iteration, void *a, void *b, void *c, void *x, void *d_LearningRate,
                                void *baseLearningRate, void *flag){

    cudaCalcGoldenSectionKernal<<<1,1>>>(iteration, (double*)a, (double*)b, (double*)c, (double*)x,
            (double*)d_LearningRate, (double*)baseLearningRate, (double*)flag);
    CUDA_CHECK(cudaGetLastError());
}

/*-------------------------------------------------------------------------------------------------------------*/

void EXEC_cudaSetParameter(void *a, void *b){
    cudaSetParameterKernal<<<1,1>>>((double*)a, (double*)b);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaPrintTransform(void *d_Transform){
    cudaPrintTransformKernal<<<1,1>>>((double*)d_Transform);
    CUDA_CHECK(cudaGetLastError());
}

/*-------------------------------------------------------------------------------------------------------------*/

int EXEC_GoldenSectionSearch(int w, int h, int spacing, int iteration, double *a, double *b, double *c, double *x,
                             double *metricb, double *metricx, double *d_LearningRate, void *d_FixedImage, void *d_MovingImage,
                             void *d_InitialTransform, void *d_AffineTransform, void *d_VirtualPointSet,
                             int NumberOfDomainPoints, int pointsInWidth, int pointsInHeightCeil, void *d_DerivativeResult,
                             void *d_ValueResult, void *d_Gradient, void *t_Value, void *t_ValidPointCount,
                             void *baseLearningRate, void *flag){

    /*-------------------------------------------------------------------------------------------------------------*/

    int metricEvaluations = 0;
    for(int searchIteration = iteration; ; ++searchIteration)
    {
        EXEC_cudaCalcGoldenSection(searchIteration, a, b, c, x, d_LearningRate, baseLearningRate, flag);

        double f;
        cudaMemcpy(&f, flag, sizeof(double)*1, cudaMemcpyDeviceToHost);
        if(f == 1){
            return metricEvaluations;
        }

        /*-------------------------------------------------------------------------------------------------------------*/
        EXEC_cudaModifyGradient(d_Gradient, d_LearningRate);
        EXEC_cudaUpdateTransform(d_AffineTransform, d_Gradient);

        EXEC_cudaCalculateAffine(w, h, spacing, d_FixedImage, d_MovingImage, d_InitialTransform, d_AffineTransform,
                                 d_VirtualPointSet, false, NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil,
                                 d_DerivativeResult, d_ValueResult);
        EXEC_cudaProcessValue(w, h, spacing, d_ValueResult, NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil,
                              metricx, t_Value, t_ValidPointCount);
        metricEvaluations++;

        EXEC_cudaResetTransform(d_AffineTransform, d_Gradient);
        EXEC_cudaResetGradient(d_Gradient, d_LearningRate);
        /*-------------------------------------------------------------------------------------------------------------*/

        if (searchIteration == 0){
            EXEC_cudaSetParameter(d_LearningRate, b);

            EXEC_cudaModifyGradient(d_Gradient, d_LearningRate);
            EXEC_cudaUpdateTransform(d_AffineTransform, d_Gradient);

            EXEC_cudaCalculateAffine(w, h, spacing, d_FixedImage, d_MovingImage, d_InitialTransform,  d_AffineTransform,
                                     d_VirtualPointSet, false, NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil,
                                     d_DerivativeResult, d_ValueResult);
            EXEC_cudaProcessValue(w, h, spacing, d_ValueResult, NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil,
                                  metricb, t_Value, t_ValidPointCount);
            metricEvaluations++;

            EXEC_cudaResetTransform(d_AffineTransform, d_Gradient);
            EXEC_cudaResetGradient(d_Gradient, d_LearningRate);

            EXEC_cudaSetParameter(d_LearningRate, baseLearningRate);
        }

        // Keep the original golden-section update order; only replace the
        // recursive host call and per-depth cudaMalloc/cudaFree with reuse.
        EXEC_cudaFindGoldenSection(a, b, c, x, metricb, metricx);
    }
}

/*-------------------------------------------------------------------------------------------------------------*/

void EXEC_cudaResampleFixedImage(int w, int h, int spacing, void *d_FixedImage, void *d_FixedInverseField, double offset_x,
                                 double offset_y, void *d_ResampledFixedImage){

    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w / spacing, h / spacing);
    cudaResampleFixedImageKernal<<<grid,block>>>(w, h, spacing, (float*)d_FixedImage, (CudaFieldValueType*)d_FixedInverseField,
                                                 offset_x, offset_y, (float*)d_ResampledFixedImage);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaResampleMovingImage(int w, int h, int spacing, void *d_MovingImage, void *d_MovingInverseField,
                                  void *d_AffineTransform, void *d_InitialTransform, double offset_x, double offset_y,
                                  void *d_ResampledMovingImage){

    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w / spacing, h / spacing);
    cudaResampleMovingImageKernal<<<grid,block>>>(w, h, spacing, (float*)d_MovingImage, (double*)d_InitialTransform,
                                                  (double*)d_AffineTransform, (CudaFieldValueType*)d_MovingInverseField, offset_x,
                                                  offset_y, (float*)d_ResampledMovingImage);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaCalculateSyn(int w, int h, int spacing, void *d_FixedImage, void *d_MovingImage, void *d_FixedDerivative,
                           void *d_MovingDerivative, void *d_ValueResult, double robustThreshold){

    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    constexpr int radius = 4;
    size_t sharedBytes = 2 * (block.x + radius * 2) * (block.y + radius * 2) * sizeof(float);
    cudaCalculateSynKernal<<<grid,block,sharedBytes>>>(w, h, spacing, (float*)d_FixedImage, (float*)d_MovingImage,
                                    (CudaFieldValueType*)d_FixedDerivative, (CudaFieldValueType*)d_MovingDerivative,
                                    (double*)d_ValueResult, robustThreshold);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaStoreCurrentValue(int w, int h, void *d_ValueResult, int currentIteration, int minIteration,
                                void *d_EnergyValues, void *d_TotalEnergy, void *t_Value){

    //
    dim3 block(256);
    dim3 grid((h + block.x - 1) / block.x);
    cudaStoreRowValueKernal<<<grid,block>>>(w, h, (double*)d_ValueResult, (double*)t_Value);
    CUDA_CHECK(cudaGetLastError());
    cudaStoreColValueKernal<<<1,1>>>(w, h, (double*)t_Value, currentIteration, minIteration,
                                     (double*)d_EnergyValues, (double*)d_TotalEnergy);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaSmoothField(int w, int h, void *d_Derivative, void *t_Derivative){
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaSmoothByRowKernal<<<grid,block>>>(w, h, (CudaFieldValueType*)d_Derivative, (CudaFieldValueType*)t_Derivative);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    cudaSmoothByColKernal<<<grid,block>>>(w, h, (CudaFieldValueType*)t_Derivative, (CudaFieldValueType*)d_Derivative);
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    cudaCheckBoundaryKernal<<<grid,block>>>(w, h, (CudaFieldValueType*)d_Derivative);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaScaleField(int w, int h, int spacing, void *d_Derivative, void *d_Scale, void *t_Scale, double maxStep){
    dim3 rowBlock(256);
    dim3 rowGrid((h + rowBlock.x - 1) / rowBlock.x);
    cudaGetRowNormKernal<<<rowGrid,rowBlock>>>(w, h, spacing, (CudaFieldValueType*)d_Derivative, (double*)t_Scale);
    CUDA_CHECK(cudaGetLastError());
    cudaGetColNormKernal<<<1,1>>>(w, h, (double*)t_Scale, (double*)d_Scale, maxStep);
    CUDA_CHECK(cudaGetLastError());
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaScaleFieldKernal<<<grid,block>>>(w, h, (CudaFieldValueType*)d_Derivative, (double*)d_Scale);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaComposeField(int w, int h, int spacing, void *d_DisplacementField, void *d_WarpingField, double offset_x,
                           double offset_y, void *t_OutputField){
    //
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaComposeFieldKernal<<<grid,block>>>(w, h, spacing, (CudaFieldValueType*)d_DisplacementField, (CudaFieldValueType*)d_WarpingField, offset_x, offset_y,
                                    (CudaFieldValueType*)t_OutputField);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaInverseWithoutEstimate(int w, int h, int spacing, void *t_ComposeField, void *t_ScaledNormImage,
                                     void *d_MeanErrorNorm, void *d_MaxErrorNorm, void *t_MeanErrorNorm, void *t_MaxErrorNorm){

    //
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaCalcScaledNormImageKernal<<<grid,block>>>(w, h, spacing, (CudaFieldValueType*)t_ComposeField, (CudaFieldValueType*)t_ScaledNormImage);
    CUDA_CHECK(cudaGetLastError());
    dim3 rowBlock(256);
    dim3 rowGrid((h + rowBlock.x - 1) / rowBlock.x);
    cudaGetRowMeanMaxErrorNormKernal<<<rowGrid,rowBlock>>>(w, h, (CudaFieldValueType*)t_ScaledNormImage, (double*)t_MeanErrorNorm,
                                              (double*)t_MaxErrorNorm);
    CUDA_CHECK(cudaGetLastError());
    cudaGetColMeanMaxErrorNormKernal<<<1,1>>>(w, h, (double*)t_MeanErrorNorm, (double*)t_MaxErrorNorm,
                                              (double*)d_MeanErrorNorm, (double*)d_MaxErrorNorm);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaInverseWithEstimate(int w, int h, void *t_ComposeField, void *t_ScaledNormImage, double epsilon,
                                  void *d_MaxErrorNorm, void *d_OutputField){
    //
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaInverseFieldKernal<<<grid,block>>>(w, h, (CudaFieldValueType*)t_ComposeField, (CudaFieldValueType*)t_ScaledNormImage, epsilon,
                                    (double*)d_MaxErrorNorm, (CudaFieldValueType*)d_OutputField);
    CUDA_CHECK(cudaGetLastError());
}

/*-------------------------------------------------------------------------------------------------------------*/

void EXEC_cudaCopyDisplacementField(int w, int h, void *d_DisplacementField, void *o_DisplacementField,
                                    double maxDisplacement){
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaCopyDisplacementFieldKernal<<<grid,block>>>(w, h, (CudaFieldValueType*)d_DisplacementField, (CudaFieldValueType*)o_DisplacementField,
                                             maxDisplacement);
    CUDA_CHECK(cudaGetLastError());
}

/*-------------------------------------------------------------------------------------------------------------*/

int EXEC_cudaInverseField(int w, int h, int spacing, void *d_Field, void *d_InverseField, double offset_x, double offset_y,
                          void *d_OutputField, void *d_MeanErrorNorm, void *d_MaxErrorNorm, void *t_ComposeField,
                          void *t_ScaledNormImage, void *t_MeanErrorNorm, void *t_MaxErrorNorm){

    int iteration = 0;
    int inverseIterationCount = 0;
    double m_MeanErrorNorm = 1.0;
    double m_MaxErrorNorm = 1.0;
    double epsilon = 0.5;

    EXEC_cudaCopyDisplacementField(w, h, d_OutputField, d_InverseField, 0);

    while(iteration++ < 20 && m_MaxErrorNorm > 0.1 && m_MeanErrorNorm > 0.001){
        inverseIterationCount++;
        EXEC_cudaComposeField(w, h, spacing, d_Field, d_OutputField, offset_x, offset_y, t_ComposeField);
        EXEC_cudaInverseWithoutEstimate(w, h, spacing, t_ComposeField, t_ScaledNormImage, d_MeanErrorNorm, d_MaxErrorNorm,
                                        t_MeanErrorNorm, t_MaxErrorNorm);
        cudaMemcpy(&m_MeanErrorNorm, d_MeanErrorNorm, sizeof(double)*1, cudaMemcpyDeviceToHost);
        cudaMemcpy(&m_MaxErrorNorm, d_MaxErrorNorm, sizeof(double)*1, cudaMemcpyDeviceToHost);
        if(iteration == 1){epsilon = 0.75;}
        else{epsilon = 0.5;}
        EXEC_cudaInverseWithEstimate(w, h, t_ComposeField, t_ScaledNormImage, epsilon, d_MaxErrorNorm, d_OutputField);
    }
    return inverseIterationCount;
}

/*-------------------------------------------------------------------------------------------------------------*/

void EXEC_cudaInitDisplacementField(int w, int h, void *d_DisplacementField){
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaInitDisplacementFieldKernal<<<grid,block>>>(w, h, (CudaFieldValueType*)d_DisplacementField);
    CUDA_CHECK(cudaGetLastError());
}

void EXEC_cudaAdaptDisplacementField(int w, int h, int spacingOld, int spacingNew, void *d_OldField, double offsetOld_x,
                                     double offsetOld_y, double offsetNew_x, double offsetNew_y, void *d_NewField){
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w / spacingNew, h / spacingNew);
    cudaAdaptDisplacementFieldKernal<<<grid,block>>>(w, h, spacingOld, spacingNew, (CudaFieldValueType*)d_OldField, offsetOld_x, offsetOld_y,
                                                     offsetNew_x, offsetNew_y, (CudaFieldValueType*)d_NewField);
    CUDA_CHECK(cudaGetLastError());
    // cudaDeviceSynchronize();
    // printf("EXEC_cudaAdaptDisplacementField after kernal function : %s\n",cudaGetErrorString(cudaGetLastError()));
}

void EXEC_cudaApplyTransform(int w, int h, void *d_OrigImage, void *d_InitialTransform, void *d_AffineTransform,
                             void *d_OutputFieldFinal, void *d_OutputInverseFieldFinal, int outputDirection,
                             void *d_ResultImage, void *d_TransformMap){

    //
    dim3 block = CudaBlock2D();
    dim3 grid = CudaGrid2D(w, h);
    cudaApplyTransformKernal<<<grid,block>>>(w, h, (float*)d_OrigImage, (double*)d_InitialTransform,
                                      (double*)d_AffineTransform, (CudaFieldValueType*)d_OutputFieldFinal,
                                      (CudaFieldValueType*)d_OutputInverseFieldFinal, outputDirection,
                                      (int*)d_ResultImage, (float*)d_TransformMap);
    CUDA_CHECK(cudaGetLastError());
}

/*-------------------------------------------------------------------------------------------------------------*/

void cudaStartAffinePerLevel(int w, int h, int spacing, float *d_FixedImage, float *d_MovingImage, double *d_InitialTransform,
                             double *d_AffineTransform, double offset_x, double offset_y, int maxIteration, int minIteration,
                             bool profile, const char *profileLabel){

    int pointsInWidth = ceil((w - offset_x) / spacing);
    int pointsInHeight = ceil((h - offset_y) / spacing);

    double maxPixel_x = offset_x + (pointsInWidth - 1) * spacing;
    double maxPixel_y = offset_y + (pointsInHeight - 1) * spacing;

    /*-------------------------------------------------------------------------------------------------------------*/

    double temp_Scales[6];  // 前4个值是Matrix,后2个值是Offset
    temp_Scales[0] = 0;  temp_Scales[1] = 0;
    temp_Scales[2] = 0;  temp_Scales[3] = 0;
    temp_Scales[4] = 0;  temp_Scales[5] = 0;

    double *d_Scales;
    cudaMalloc((void**) &d_Scales,  sizeof(double) * 6);
    cudaMemcpy(d_Scales, temp_Scales, sizeof(double) * 6, cudaMemcpyHostToDevice);

    /*-------------------------------------------------------------------------------------------------------------*/

    int NumberOfDomainPoints = ceil((pointsInWidth * pointsInHeight) / 5.0);
    int pointsInHeightCeil = ceil(pointsInHeight / 5.0);
    int pointCount = 0;
    double* m_VirtualPointSet = new double[NumberOfDomainPoints * 2];   // NumberOfDomainPoints:1568-6336-25344

    // for(double height = offset_y; height < h; height += spacing){
    //     for(double width = offset_x; width < w; width += spacing){
    //         if(pointCount % 5 == 0){
    //             m_VirtualPointSet[pointCount * 2 / 5 + 0] = width + getGaussianRandomValue() * spacing / 3;
    //             m_VirtualPointSet[pointCount * 2 / 5 + 1] = height + getGaussianRandomValue() * spacing / 3;
    //         }
    //         pointCount += 1;
    //     }
    // }

//     std::cout << " pointsInWidth:" << pointsInWidth << " pointsInHeight:" << pointsInHeight
//               << " NumberOfDomainPoints:" << NumberOfDomainPoints << " pointsInHeightCeil:" << pointsInHeightCeil << std::endl;

    int xx = 0;
    for(double height = offset_y; height < h; height += spacing){
        for(double width = offset_x; width < w; width += spacing){
            if(pointCount % 5 == 0){
                m_VirtualPointSet[pointCount * 2 / 5 + 0] = width;
                m_VirtualPointSet[pointCount * 2 / 5 + 1] = height;
                xx += 1;
            }
            pointCount += 1;
        }
    }

    double *d_VirtualPointSet;
    cudaMalloc((void **) &d_VirtualPointSet, sizeof(double) * NumberOfDomainPoints * 2);
    cudaMemcpy(d_VirtualPointSet, m_VirtualPointSet, sizeof(double) * NumberOfDomainPoints * 2, cudaMemcpyHostToDevice);

    delete []m_VirtualPointSet;

    /*-------------------------------------------------------------------------------------------------------------*/

    CudaAffineMetricValueType *d_DerivativeResult;
    cudaMalloc((void **) &d_DerivativeResult, sizeof(CudaAffineMetricValueType) * NumberOfDomainPoints * 6);

    CudaAffineMetricValueType *d_ValueResult;
    cudaMalloc((void **) &d_ValueResult, sizeof(CudaAffineMetricValueType) * NumberOfDomainPoints * 1);

    double *t_Gradient;
    cudaMalloc((void**) &t_Gradient, sizeof(double) * pointsInHeightCeil * 6);

    CudaAffineMetricValueType *t_Value;
    cudaMalloc((void**) &t_Value, sizeof(CudaAffineMetricValueType) * pointsInHeightCeil * 1);

    CudaAffineMetricValueType *t_ValidPointCount;
    cudaMalloc((void**) &t_ValidPointCount, sizeof(CudaAffineMetricValueType) * pointsInHeightCeil * 1);

    /*-------------------------------------------------------------------------------------------------------------*/

    double *d_Gradient;
    cudaMalloc((void**) &d_Gradient,  sizeof(double) * 6);

    double *d_LearningRate;
    cudaMalloc((void**) &d_LearningRate, sizeof(double) * 1);

    double *d_Value;
    cudaMalloc((void**) &d_Value, sizeof(double) * 1);

    double *a;
    cudaMalloc((void**) &a, sizeof(double) * 1);

    double *b;
    cudaMalloc((void**) &b, sizeof(double) * 1);

    double *c;
    cudaMalloc((void**) &c, sizeof(double) * 1);

    double *x;
    cudaMalloc((void**) &x, sizeof(double) * 1);

    double *metricb;
    cudaMalloc((void**) &metricb, sizeof(double) * 1);

    double *metricx;
    cudaMalloc((void**) &metricx, sizeof(double) * 1);

    double *baseLearningRate;
    cudaMalloc((void**) &baseLearningRate, sizeof(double) * 1);

    double *goldenSectionFlag;
    cudaMalloc((void**) &goldenSectionFlag, sizeof(double) * 1);

    /*-------------------------------------------------------------------------------------------------------------*/

    double *d_EnergyValues;
    cudaMalloc((void**) &d_EnergyValues, sizeof(double) * minIteration);

    double *d_TotalEnergy;
    cudaMalloc((void**) &d_TotalEnergy, sizeof(double) * 1);
    double m_TotalEnergy = 0;
    cudaMemcpy(d_TotalEnergy, &m_TotalEnergy, sizeof(double) * 1, cudaMemcpyHostToDevice);

    double *d_ConvergenceValue;
    cudaMalloc((void**) &d_ConvergenceValue, sizeof(double) * 1);

    /*-------------------------------------------------------------------------------------------------------------*/

    double *d_DeltaLattice;
    cudaMalloc((void**) &d_DeltaLattice,  sizeof(double) * minIteration * 2);

    double *d_OmegaLattice;
    cudaMalloc((void**) &d_OmegaLattice,  sizeof(double) * minIteration * 2);

    /*-------------------------------------------------------------------------------------------------------------*/

    EXEC_cudaEstimateScales(d_InitialTransform, d_AffineTransform, offset_x, offset_y, maxPixel_x, maxPixel_y, d_Scales);

    /*-------------------------------------------------------------------------------------------------------------*/

    double profileCalculateMs = 0.0;
    double profileProcessMs = 0.0;
    double profileConvergenceMs = 0.0;
    double profileSearchMs = 0.0;
    int profileIterations = 0;
    int profileSearchEvaluations = 0;

    for(int currentIteration = 0; currentIteration < maxIteration; currentIteration++){

//         std::cout << "Affine iteration:" << currentIteration << std::endl;

        LARGE_INTEGER t1, t2, t3, t4, t5, t6, tc;
        QueryPerformanceFrequency(&tc);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t1);

        EXEC_cudaCalculateAffine(w, h, spacing, d_FixedImage, d_MovingImage, d_InitialTransform,  d_AffineTransform,
                                 d_VirtualPointSet, true, NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil,
                                 d_DerivativeResult, d_ValueResult);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t2);
        profileCalculateMs += ((t2.QuadPart - t1.QuadPart) * 1000.0 / tc.QuadPart);

        // std::cout << " EXEC_cudaCalculateAffine:" << ((t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        EXEC_cudaProcessGradient(w, h, spacing, d_DerivativeResult, d_ValueResult, d_Scales, NumberOfDomainPoints,
                                 pointsInWidth, pointsInHeightCeil, currentIteration, minIteration, d_Gradient, d_Value,
                                 d_EnergyValues, d_TotalEnergy, t_Gradient, t_Value, t_ValidPointCount);
        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t3);
        profileProcessMs += ((t3.QuadPart - t2.QuadPart) * 1000.0 / tc.QuadPart);

        // std::cout << " EXEC_cudaProcessGradient:" << ((t3.QuadPart - t2.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        /*-------------------------------------------------------------------------------------------------------------*/

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t4);

        if(currentIteration >= minIteration - 1){
            EXEC_cudaGetConvergenceValue(d_EnergyValues, d_TotalEnergy, currentIteration, minIteration, d_ConvergenceValue,
                                         d_DeltaLattice, d_OmegaLattice);
            double m_ConvergenceValue;
            cudaMemcpy(&m_ConvergenceValue, d_ConvergenceValue, sizeof(double)*1, cudaMemcpyDeviceToHost);
            // std::cout << "ConvergenceValue:" << m_ConvergenceValue << std::endl;
            if(m_ConvergenceValue < 1e-6){
                profileIterations++;
                break;
            }
        }

        /*-------------------------------------------------------------------------------------------------------------*/

        if (currentIteration == 0){
            EXEC_cudaEstimateLearningRate(d_InitialTransform, d_AffineTransform, d_Gradient,
                                          offset_x, offset_y, maxPixel_x, maxPixel_y, d_LearningRate);
            cudaDeviceSynchronize();
        }

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t5);
        profileConvergenceMs += ((t5.QuadPart - t4.QuadPart) * 1000.0 / tc.QuadPart);

        // std::cout << " EXEC_cudaEstimateLearningRate / EXEC_cudaGetConvergenceValue:"
        //           << ((t5.QuadPart - t4.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        /*-------------------------------------------------------------------------------------------------------------*/

        EXEC_cudaSetGoldenSection(d_LearningRate, d_Gradient, a, b, c);

        profileSearchEvaluations += EXEC_GoldenSectionSearch(w, h, spacing, 0, a, b, c, x, metricb, metricx, d_LearningRate, d_FixedImage, d_MovingImage,
            d_InitialTransform, d_AffineTransform, d_VirtualPointSet, NumberOfDomainPoints, pointsInWidth, pointsInHeightCeil,
            d_DerivativeResult, d_ValueResult, d_Gradient, t_Value, t_ValidPointCount, baseLearningRate,
            goldenSectionFlag);

        EXEC_cudaModifyGradient(d_Gradient, d_LearningRate);
        EXEC_cudaUpdateTransform(d_AffineTransform, d_Gradient);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t6);
        profileSearchMs += ((t6.QuadPart - t5.QuadPart) * 1000.0 / tc.QuadPart);
        profileIterations++;

        double m_LearningRate;
        cudaMemcpy(&m_LearningRate, d_LearningRate, sizeof(double)*1, cudaMemcpyDeviceToHost);
        // Reuse the existing readback to finish a stationary/rejected affine
        // step. This is exact-zero safety, not a residual-based early-stop rule.
        if(m_LearningRate == 0.0){break;}
//         std::cout << "LearningRate(afterSearch):" << m_LearningRate << std::endl;

        // std::cout << " EXEC_GoldenSectionSearch:" << ((t6.QuadPart - t5.QuadPart)*1.0 / tc.QuadPart) << std::endl;
//         std::cout << "---------------------------------------------------------------------------------" << std::endl;
    }

    if(profile){
        std::cout << "[cudaRegistration.profile] native." << profileLabel
                  << ".iterations=" << profileIterations
                  << " search_evaluations=" << profileSearchEvaluations
                  << " calculate_ms=" << profileCalculateMs
                  << " process_ms=" << profileProcessMs
                  << " convergence_ms=" << profileConvergenceMs
                  << " search_ms=" << profileSearchMs
                  << std::endl;
    }

    cudaFree(d_Scales);
    cudaFree(d_Gradient);
    cudaFree(d_VirtualPointSet);
    cudaFree(d_DerivativeResult);
    cudaFree(d_ValueResult);
    cudaFree(t_Gradient);
    cudaFree(t_Value);
    cudaFree(t_ValidPointCount);
    cudaFree(d_LearningRate);
    cudaFree(d_Value);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(x);
    cudaFree(metricb);
    cudaFree(metricx);
    cudaFree(baseLearningRate);
    cudaFree(goldenSectionFlag);
    cudaFree(d_EnergyValues);
    cudaFree(d_TotalEnergy);
    cudaFree(d_ConvergenceValue);
    cudaFree(d_DeltaLattice);
    cudaFree(d_OmegaLattice);
}

void cudaStartSynPerLevel(int w, int h, int spacing, float *d_FixedImage, float *d_MovingImage, double *d_InitialTransform,
                          double *d_AffineTransform, CudaFieldValueType *d_FixedField, CudaFieldValueType *d_FixedInverseField, CudaFieldValueType *d_MovingField,
                          CudaFieldValueType *d_MovingInverseField, double offset_x, double offset_y, int maxIteration,
                          int minIteration, double robustThreshold, double synMaxStep, double synMaxDisplacement,
                          bool profile, const char *profileLabel){

    double profileResampleMs = 0.0;
    double profileCalculateMs = 0.0;
    double profileSmoothMs = 0.0;
    double profileScaleMs = 0.0;
    double profileComposeMs = 0.0;
    double profileInverseMs = 0.0;
    double profileConvergenceMs = 0.0;
    int profileIterations = 0;
    int profileInverseCalls = 0;
    int profileInverseIterations = 0;

    float *d_ResampledFixedImage;
    cudaMalloc((void**) &d_ResampledFixedImage, sizeof(float) * (w / spacing) * (h / spacing));

    float *d_ResampledMovingImage;
    cudaMalloc((void**) &d_ResampledMovingImage, sizeof(float) * (w / spacing) * (h / spacing));

    CudaFieldValueType *d_FixedDerivative;
    cudaMalloc((void**) &d_FixedDerivative, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *d_MovingDerivative;
    cudaMalloc((void**) &d_MovingDerivative, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *t_FixedDerivative;
    cudaMalloc((void**) &t_FixedDerivative, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *t_MovingDerivative;
    cudaMalloc((void**) &t_MovingDerivative, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    double *d_ValueResult;
    cudaMalloc((void**) &d_ValueResult, sizeof(double) * (w / spacing) * (h / spacing));

    /*-------------------------------------------------------------------------------------------------------------*/

    double *t_Value;
    cudaMalloc((void**) &t_Value, sizeof(double) * (h / spacing));

    double *d_EnergyValues;
    cudaMalloc((void**) &d_EnergyValues, sizeof(double) * minIteration);

    double *d_TotalEnergy;
    cudaMalloc((void**) &d_TotalEnergy, sizeof(double) * 1);
    double m_TotalEnergy = 0;
    cudaMemcpy(d_TotalEnergy, &m_TotalEnergy, sizeof(double) * 1, cudaMemcpyHostToDevice);

    double *d_ConvergenceValue;
    cudaMalloc((void**) &d_ConvergenceValue, sizeof(double) * 1);

    double *d_DeltaLattice;
    cudaMalloc((void**) &d_DeltaLattice,  sizeof(double) * minIteration * 2);

    double *d_OmegaLattice;
    cudaMalloc((void**) &d_OmegaLattice,  sizeof(double) * minIteration * 2);

    /*-------------------------------------------------------------------------------------------------------------*/

    double *t_Scale;
    cudaMalloc((void**) &t_Scale, sizeof(double) * (h / spacing));

    double *d_Scale;
    cudaMalloc((void**) &d_Scale, sizeof(double) * 1);

    /*-------------------------------------------------------------------------------------------------------------*/

    CudaFieldValueType *t_FixedField;
    cudaMalloc((void**) &t_FixedField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *t_MovingField;
    cudaMalloc((void**) &t_MovingField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    /*-------------------------------------------------------------------------------------------------------------*/

    double *d_MeanErrorNorm;
    cudaMalloc((void**) &d_MeanErrorNorm, sizeof(double) * 1);

    double *d_MaxErrorNorm;
    cudaMalloc((void**) &d_MaxErrorNorm, sizeof(double) * 1);

    double *t_MeanErrorNorm;
    cudaMalloc((void**) &t_MeanErrorNorm, sizeof(double) * (h / spacing));

    double *t_MaxErrorNorm;
    cudaMalloc((void**) &t_MaxErrorNorm, sizeof(double) * (h / spacing));

    CudaFieldValueType *t_ComposeField;
    cudaMalloc((void**) &t_ComposeField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *t_ScaledNormImage;
    cudaMalloc((void**) &t_ScaledNormImage, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 1);

    /*-------------------------------------------------------------------------------------------------------------*/

    CudaFieldValueType *o_FixedField;
    cudaMalloc((void**) &o_FixedField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *o_FixedInverseField;
    cudaMalloc((void**) &o_FixedInverseField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *o_MovingField;
    cudaMalloc((void**) &o_MovingField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    CudaFieldValueType *o_MovingInverseField;
    cudaMalloc((void**) &o_MovingInverseField, sizeof(CudaFieldValueType) * (w / spacing) * (h / spacing) * 2);

    /*-------------------------------------------------------------------------------------------------------------*/

    for(int currentIteration = 0; currentIteration < maxIteration; currentIteration++){

//         std::cout << "SyN iteration:" << currentIteration << std::endl;

        LARGE_INTEGER t1, t2, t3, t4, t5, t6, tc;
        QueryPerformanceFrequency(&tc);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t1);

        // 1.resample fixedImage (only according to "spacing","offset","fixedField")
        EXEC_cudaResampleFixedImage(w, h, spacing, d_FixedImage, d_FixedInverseField, offset_x, offset_y, d_ResampledFixedImage);

        // 2.resample movingImage (according to "spacing","offset","movingField","affineTransform","initialTransform")
        EXEC_cudaResampleMovingImage(w, h, spacing, d_MovingImage, d_MovingInverseField, d_AffineTransform, d_InitialTransform,
                                     offset_x, offset_y, d_ResampledMovingImage);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t2);

        // std::cout << " EXEC_cudaResampleImage:" << ((t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        /*-------------------------------------------------------------------------------------------------------------*/

        // 3.getValueAndDerivative
        // d_MovingDerivative is get from d_ResampledFixedImage, d_FixedDerivative is get from d_ResampledMovingImage
        EXEC_cudaCalculateSyn(w / spacing, h / spacing, spacing, d_ResampledFixedImage, d_ResampledMovingImage,
                              d_FixedDerivative, d_MovingDerivative, d_ValueResult, robustThreshold);
        EXEC_cudaStoreCurrentValue(w / spacing, h / spacing, d_ValueResult, currentIteration, minIteration, d_EnergyValues,
                                   d_TotalEnergy, t_Value);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t3);

        // std::cout << " EXEC_cudaCalculateSyn:" << ((t3.QuadPart - t2.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        /*-------------------------------------------------------------------------------------------------------------*/

        // 4.gaussianSmooth the field
        EXEC_cudaSmoothField(w / spacing, h / spacing, d_FixedDerivative, t_FixedDerivative);
        EXEC_cudaSmoothField(w / spacing, h / spacing, d_MovingDerivative, t_MovingDerivative);
        LARGE_INTEGER tSmooth;
        if(profile){
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&tSmooth);
        }

        // 5.scaleUpdateField
        EXEC_cudaScaleField(w / spacing, h / spacing, spacing, d_FixedDerivative, d_Scale, t_Scale, synMaxStep);
        EXEC_cudaScaleField(w / spacing, h / spacing, spacing, d_MovingDerivative, d_Scale, t_Scale, synMaxStep);
        LARGE_INTEGER tScale;
        if(profile){
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&tScale);
        }

        // 6.ComposeDisplacementFields
        EXEC_cudaComposeField(w / spacing, h / spacing, spacing, d_FixedDerivative, d_FixedField, offset_x, offset_y,
                              t_FixedField);
        EXEC_cudaComposeField(w / spacing, h / spacing, spacing, d_MovingDerivative, d_MovingField, offset_x, offset_y,
                              t_MovingField);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t4);

        // std::cout << " EXEC_cudaSmooth+Scale+Compose:" << ((t4.QuadPart - t3.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        /*-------------------------------------------------------------------------------------------------------------*/

//         std::cout << ".............................." << std::endl;
        // 7.InvertDisplacementField
        int inverseCount = 0;
        inverseCount += EXEC_cudaInverseField(w / spacing, h / spacing, spacing, t_FixedField, d_FixedInverseField, offset_x, offset_y,
                                              o_FixedInverseField, d_MeanErrorNorm, d_MaxErrorNorm, t_ComposeField, t_ScaledNormImage,
                                              t_MeanErrorNorm, t_MaxErrorNorm);
        EXEC_cudaCopyDisplacementField(w / spacing, h / spacing, d_FixedInverseField, o_FixedInverseField,
                                       synMaxDisplacement);

        cudaDeviceSynchronize();
//         std::cout << ".............................." << std::endl;

        inverseCount += EXEC_cudaInverseField(w / spacing, h / spacing, spacing, o_FixedInverseField, t_FixedField, offset_x, offset_y,
                                              o_FixedField, d_MeanErrorNorm, d_MaxErrorNorm, t_ComposeField, t_ScaledNormImage,
                                              t_MeanErrorNorm, t_MaxErrorNorm);
        EXEC_cudaCopyDisplacementField(w / spacing, h / spacing, d_FixedField, o_FixedField, synMaxDisplacement);

        cudaDeviceSynchronize();
//         std::cout << ".............................." << std::endl;

        inverseCount += EXEC_cudaInverseField(w / spacing, h / spacing, spacing, t_MovingField, d_MovingInverseField, offset_x, offset_y,
                                              o_MovingInverseField, d_MeanErrorNorm, d_MaxErrorNorm, t_ComposeField, t_ScaledNormImage,
                                              t_MeanErrorNorm, t_MaxErrorNorm);
        EXEC_cudaCopyDisplacementField(w / spacing, h / spacing, d_MovingInverseField, o_MovingInverseField,
                                       synMaxDisplacement);

        cudaDeviceSynchronize();
//         std::cout << ".............................." << std::endl;

        inverseCount += EXEC_cudaInverseField(w / spacing, h / spacing, spacing, o_MovingInverseField, t_MovingField, offset_x, offset_y,
                                              o_MovingField, d_MeanErrorNorm, d_MaxErrorNorm, t_ComposeField, t_ScaledNormImage,
                                              t_MeanErrorNorm, t_MaxErrorNorm);
        EXEC_cudaCopyDisplacementField(w / spacing, h / spacing, d_MovingField, o_MovingField, synMaxDisplacement);

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t5);

        // std::cout << " EXEC_cudaInverseField:" << ((t5.QuadPart - t4.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        /*-------------------------------------------------------------------------------------------------------------*/

        // 4.GetConvergenceValue
        if(currentIteration >= minIteration - 1){
            EXEC_cudaGetConvergenceValue(d_EnergyValues, d_TotalEnergy, currentIteration, minIteration, d_ConvergenceValue,
                                         d_DeltaLattice, d_OmegaLattice);
            double m_ConvergenceValue;
            cudaMemcpy(&m_ConvergenceValue, d_ConvergenceValue, sizeof(double)*1, cudaMemcpyDeviceToHost);
            if(m_ConvergenceValue < 1e-7){break;}
        }

        cudaDeviceSynchronize();
        QueryPerformanceCounter(&t6);

        // std::cout << " EXEC_cudaGetConvergenceValue:" << ((t6.QuadPart - t5.QuadPart)*1.0 / tc.QuadPart) << std::endl;

        if(profile){
            profileIterations++;
            profileInverseCalls += 4;
            profileInverseIterations += inverseCount;
            profileResampleMs += ((t2.QuadPart - t1.QuadPart) * 1000.0 / tc.QuadPart);
            profileCalculateMs += ((t3.QuadPart - t2.QuadPart) * 1000.0 / tc.QuadPart);
            profileSmoothMs += ((tSmooth.QuadPart - t3.QuadPart) * 1000.0 / tc.QuadPart);
            profileScaleMs += ((tScale.QuadPart - tSmooth.QuadPart) * 1000.0 / tc.QuadPart);
            profileComposeMs += ((t4.QuadPart - tScale.QuadPart) * 1000.0 / tc.QuadPart);
            profileInverseMs += ((t5.QuadPart - t4.QuadPart) * 1000.0 / tc.QuadPart);
            profileConvergenceMs += ((t6.QuadPart - t5.QuadPart) * 1000.0 / tc.QuadPart);
        }

//         std::cout << "---------------------------------------------------------------------------------" << std::endl;
    }

    if(profile){
        std::cout << "[cudaRegistration.profile] native." << profileLabel
                  << ".iterations=" << profileIterations
                  << " inverse_calls=" << profileInverseCalls
                  << " inverse_iterations=" << profileInverseIterations
                  << std::endl;
        std::cout << "[cudaRegistration.profile] native." << profileLabel
                  << ".resample_ms=" << profileResampleMs
                  << " calculate_ms=" << profileCalculateMs
                  << " smooth_ms=" << profileSmoothMs
                  << " scale_ms=" << profileScaleMs
                  << " compose_ms=" << profileComposeMs
                  << " inverse_ms=" << profileInverseMs
                  << " convergence_ms=" << profileConvergenceMs
                  << std::endl;
    }

    cudaFree(d_ResampledFixedImage);
    cudaFree(d_ResampledMovingImage);
    cudaFree(d_FixedDerivative);
    cudaFree(d_MovingDerivative);
    cudaFree(t_FixedDerivative);
    cudaFree(t_MovingDerivative);
    cudaFree(d_ValueResult);
    cudaFree(t_Value);
    cudaFree(d_EnergyValues);
    cudaFree(d_TotalEnergy);
    cudaFree(d_ConvergenceValue);
    cudaFree(d_DeltaLattice);
    cudaFree(d_OmegaLattice);
    cudaFree(d_Scale);
    cudaFree(t_Scale);
    cudaFree(t_FixedField);
    cudaFree(t_MovingField);
    cudaFree(d_MeanErrorNorm);
    cudaFree(d_MaxErrorNorm);
    cudaFree(t_ComposeField);
    cudaFree(t_ScaledNormImage);
    cudaFree(t_MeanErrorNorm);
    cudaFree(t_MaxErrorNorm);
    cudaFree(o_FixedField);
    cudaFree(o_FixedInverseField);
    cudaFree(o_MovingField);
    cudaFree(o_MovingInverseField);
}

}
