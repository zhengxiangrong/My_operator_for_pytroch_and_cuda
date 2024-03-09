#include <vector>
#include "my_dcn_im2col.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>



std::vector<at::Tensor>
cuda_forward(const at::Tensor &input,
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w)
{
    using scalar_t  = float;
    AT_ASSERTM(input.is_cuda(),"input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(),"weight must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(),"bias must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(),"offset must be a CUDA tensor");

    const int batch = input.size(0); //（2，64，128，128）
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0); //（64，64，3，3）
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(channels == channels_kernel,"input channel and kernel channel wont match. %d vs %d",channels,channels_kernel);
    AT_ASSERTM(kernel_h == kernel_h_,"input kernel height and kernel_h wont match. %d vs %d",kernel_h_,kernel_h);
    AT_ASSERTM(kernel_w == kernel_w_,"input kernel width and kernel_w wont match. %d vs %d",kernel_w_,kernel_w);
    const int height_out = (height + 2*pad_h - (dilation_h-1)*(kernel_h-1)-kernel_h)/stride_h + 1;
    const int width_out = (width + 2*pad_w - (dilation_w-1)*(kernel_w-1)-kernel_w)/stride_w + 1;

    at::Tensor output = at::zeros({batch, channels * kernel_h * kernel_w, height_out*width_out}, input.options());
    at::Tensor weight_ = weight.view({channels_out, channels_kernel * kernel_h * kernel_w});
    at::Tensor one_ = at::ones({batch, bias.sizes()[0], height_out, width_out}, input.options());
    at::Tensor one_T = at::transpose(one_, 3, 1);

    at::Tensor bias_broadCast =  at::mul(one_T ,bias);
    bias_broadCast = at::transpose(bias_broadCast, 3, 1);
    my_dcn_im2col_forward(c10::cuda::getCurrentCUDAStream(),
                            input.data_ptr<scalar_t>(),
                            offset.data_ptr<scalar_t>(),
                            batch, channels, height, width,
                            channels_out, kernel_h,kernel_w,
                            height_out, width_out, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
                            output.data_ptr<scalar_t>());

    output = at::transpose(output, 2, 1);
    weight_ = at::transpose(weight_, 1, 0);
    at::Tensor final_rt = at::matmul(output, weight_);
    final_rt = at::transpose(final_rt, 2, 1);
    final_rt = at::add(final_rt.view({batch, channels_out, height_out, width_out}), bias_broadCast);
    return {final_rt, output};
}


std::vector<at::Tensor>
cuda_backward(const at::Tensor & grad_output,
                    const at::Tensor & output,
                    const at::Tensor & input,
                    const at::Tensor & weight,
                    const at::Tensor & offset,
                    const at::Tensor & bias,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w
)
{
    using scalar_t  = float;
    AT_ASSERTM(input.is_cuda(),"input must be a CUDA tensor");
    AT_ASSERTM(weight.is_cuda(),"weight must be a CUDA tensor");
    AT_ASSERTM(bias.is_cuda(),"bias must be a CUDA tensor");
    AT_ASSERTM(offset.is_cuda(),"offset must be a CUDA tensor");

    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(channels == channels_kernel, "input channel and kernel channel wont match. %d vs %d",channels,channels_kernel);
    AT_ASSERTM(kernel_h == kernel_h_, "input kernel height and kernel_h wont match. %d vs %d",kernel_h_,kernel_h);
    AT_ASSERTM(kernel_w == kernel_w_, "input kernel width and kernel_w wont match. %d vs %d",kernel_w_,kernel_w);
    const int height_out = (height + 2 * pad_h - (dilation_h-1)*(kernel_h-1)-kernel_h) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w-1)*(kernel_w-1)-kernel_w) / stride_w + 1;

    at::Tensor grad_output_ = grad_output.view({batch, channels_out, height_out * width_out});
    grad_output_ = at::transpose(grad_output_, 2, 1);

    at::Tensor one_ = at::ones({batch * height_out * width_out}, grad_output.options());
    at::Tensor weight_ = weight.view({channels_out, channels * kernel_h * kernel_w});
    at::Tensor grad_back = at::matmul(grad_output_, weight_); // (batch, height_out * width_out, channels * kernel_h * kernel_w)
    grad_back = at::transpose(grad_back, 2, 1);

    grad_back = grad_back.view({batch, channels, kernel_h * kernel_w, height_out * width_out});
    // output = at::zeros({batch, channels * kernel_h * kernel_w, height_out*width_out}, input.options());
    at::Tensor input_grad = at::zeros({batch, channels, height, width}, input.options());
    at::Tensor offset_grad = at::zeros({batch, 2*kernel_h*kernel_w, height_out, width_out}, offset.options());
    //at::Tensor weight_grad = at::zeros({channels_out, channels, kernel_h, kernel_w}, offset.options());
    
    // grad for inputs
    my_dcn_backward_for_input(c10::cuda::getCurrentCUDAStream(),
                              grad_back.data_ptr<scalar_t>(),
                              input.data_ptr<scalar_t>(),
                              offset.data_ptr<scalar_t>(),
                              weight.data_ptr<scalar_t>(),
                              batch, channels, height, width,
                              channels_out, kernel_h, kernel_w,
                              height_out, width_out, stride_h, stride_w,
                              pad_h, pad_w, dilation_h, dilation_w,
                              input_grad.data_ptr<scalar_t>());

    // grad for offset
    my_dcn_backward_for_offset(c10::cuda::getCurrentCUDAStream(),
                              grad_back.data_ptr<scalar_t>(),
                              input.data_ptr<scalar_t>(),
                              offset.data_ptr<scalar_t>(),
                              weight.data_ptr<scalar_t>(),
                              batch, channels, height, width,
                              channels_out, kernel_h, kernel_w,
                              height_out, width_out, stride_h, stride_w,
                              pad_h, pad_w, dilation_h, dilation_w,
                              offset_grad.data_ptr<scalar_t>());

    // batch, channels * kernel_h * kernel_w, height_out*width_out
    at::Tensor output_ = output.contiguous().view({batch, channels, kernel_h * kernel_w,  height_out * width_out});
    output_ = at::transpose(output_, 3, 1);
    output_ = output_.contiguous().view({batch * height_out * width_out, channels * kernel_h * kernel_w});
    grad_output_ = grad_output_.contiguous().view({batch* height_out * width_out, channels_out});
    grad_output_ =at::transpose(grad_output_, 1, 0);
    at::Tensor weight_grad = at::matmul(grad_output_, output_);
    weight_grad = weight_grad.contiguous().view({channels_out, channels, kernel_h, kernel_w});
    at::Tensor bias_grad  = at::matmul(grad_output_, one_);
    return {input_grad, offset_grad, weight_grad, bias_grad};
}