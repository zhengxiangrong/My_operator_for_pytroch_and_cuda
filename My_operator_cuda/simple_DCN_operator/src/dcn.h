#pragma once
#include <vector>
#include "cuda/my_dcn_cuda.h"
using namespace std;

std::vector<at::Tensor>
dcn_forward(const at::Tensor & input,
                const at::Tensor & weight,
                const at::Tensor & bias,
                const at::Tensor & offset,
                const int kernel_h,
                const int kernel_w,
                const int stride_h,
                const int stride_w,
                const int pad_h,
                const int pad_w,
                const int dilation_h,
                const int dilation_w)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return cuda_forward(input, weight, bias, offset,
                    kernel_h, kernel_w, stride_h,stride_w,
                    pad_h, pad_w, dilation_h, dilation_w);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
}


std::vector<at::Tensor>
dcn_backward(const at::Tensor & grad_output,
                const at::Tensor & output,
                const at::Tensor & input,
                const at::Tensor & weight,
                const at::Tensor & offset,
                const at::Tensor & bias,
                const int kernel_h, const int kernel_w,
                const int stride_h, const int stride_w,
                const int pad_h,
                const int pad_w,
                const int dilation_h,
                const int dilation_w)
{
    if (input.type().is_cuda())
    {
#ifdef WITH_CUDA
        return cuda_backward(grad_output, output, input, weight, offset, bias, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,dilation_w);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
}
