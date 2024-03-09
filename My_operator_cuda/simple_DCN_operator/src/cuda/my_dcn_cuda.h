#pragma once
#include <torch/extension.h>
#include <ATen/div_rtn.h>
using namespace std;
std::vector<at::Tensor>
cuda_forward(const at::Tensor & input,
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
                    const int dilation_w);


std::vector<at::Tensor>
cuda_backward(const at::Tensor & grad_output,
                    const at::Tensor & output,
                    const at::Tensor & input,
                    const at::Tensor & weight,
                    const at::Tensor & offset,
                    const at::Tensor & bias,
                    const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,
                    const int pad_h, const int pad_w, const int dilation_h, const int dilation_w);