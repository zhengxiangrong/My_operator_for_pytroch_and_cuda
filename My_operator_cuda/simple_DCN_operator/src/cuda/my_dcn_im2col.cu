#include "my_dcn_im2col.h"
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#define align_threads (1024)
#define CUDA_KERNEL_LOOP(index,n) \
      for (int index; index < n; index += blockDim.x*gridDim.x)

__device__ float bilinear_interpolation(const float inh, const float inw, const int height, const int width, const float *input)
{
    const int h_floor = floor(inh);
    const int w_floor = floor(inw);
    const int h_upper = h_floor + 1;
    const int w_upper = w_floor + 1;
    float delta_y = (inh - h_floor);
    float delta_x = (inw - w_floor);
    float left_up_area = delta_y*delta_x;
    float right_up_area = delta_y - left_up_area;
    float left_bottom_area = delta_x - left_up_area;
    float right_bottom_area = (1 - delta_x) - right_up_area;

    // offset 后所对应的方格的四个顶点
    float left_up_point = *(input + h_floor * width + w_floor);
    float right_up_point = *(input + h_floor * width + w_upper);
    float left_bottom_point = *(input + h_upper * width + w_floor);
    float right_bottom_point = *(input + h_upper * width + w_upper);

    float st = left_up_point * right_bottom_area + right_up_point * left_bottom_area + left_bottom_point * right_up_area + right_bottom_point * left_up_area;
    return st;
}

__device__ float grad_offset_weight(const float inh, const float inw, const int height, const int width, const float * input, const int weight_kind)
{
    // weight_kind = 0 时为对offset_x 求导， weight_kind = 1 时为对offset_y 求导
    const int h_floor = floor(inh);
    const int w_floor = floor(inw);
    const int h_upper = h_floor + 1;
    const int w_upper = w_floor + 1;
    float delta_y = (inh - h_floor);
    float delta_x = (inw - w_floor);
    float left_up_area = delta_y*delta_x;
    float right_up_area = delta_y - left_up_area;
    float left_bottom_area = delta_x - left_up_area;
    float right_bottom_area = (1 - delta_x) - right_up_area;
    float weight = 0;
    if (weight_kind ==1 ){
        if (h_floor>0 &&  h_floor <height && 0<w_floor &&  w_floor<width ){
            float up_left_point = *(input + h_floor * width + w_floor);
            weight += -(1-delta_y) * up_left_point;
        }
        if (h_floor>0 &&  h_floor <height && 0<w_upper &&  w_upper<width ){
            float up_right_point = *(input + h_floor * width + w_upper);
            weight += (1-delta_y) * up_right_point;
        }
        if (h_upper>0 &&  h_upper <height && 0<w_floor &&  w_floor<width ){
            float bottom_left_point = *(input + h_upper * width + w_floor);
            weight += -delta_y * bottom_left_point;
        }
        if (h_upper>0 &&  h_upper <height && 0<w_upper &&  w_upper<width ){
            float bottom_right_point = *(input + h_upper * width + w_upper);
            weight += delta_y * bottom_right_point;
        }
    }


    if (weight_kind ==0 ){
        if (h_floor>0 &&  h_floor <height && 0<w_floor &&  w_floor<width ){
            float up_left_point = *(input + h_floor * width + w_floor);
            weight += -(1-delta_x) * up_left_point;
        }
        if (h_floor>0 &&  h_floor <height && 0<w_upper &&  w_upper<width ){
            float up_right_point = *(input + h_floor * width + w_upper);
            weight += -delta_x * up_right_point;
        }
        if (h_upper>0 &&  h_upper <height && 0<w_floor &&  w_floor<width ){
            float bottom_left_point = *(input + h_upper * width + w_floor);
            weight += (1-delta_x) * bottom_left_point;
        }
        if (h_upper>0 &&  h_upper <height && 0<w_upper &&  w_upper<width ){
            float bottom_right_point = *(input + h_upper * width + w_upper);
            weight += delta_x * bottom_right_point;
        }
    }
    return weight;
}


__device__ float input_grad_weight(const float inh, const float inw, const int weight_kind){
    if (weight_kind == 0) // left_up
    {
        const int h_floor = floor(inh);
        const int w_floor = floor(inw);
        const int h_upper = h_floor + 1;
        const int w_upper = w_floor + 1;
        float delta_y = (inh - h_floor);
        float delta_x = (inw - w_floor);
        float left_up_area = delta_y*delta_x;
        return left_up_area;
    } else if(weight_kind == 1) // right_up
    {
        const int h_floor = floor(inh);
        const int w_floor = floor(inw);
        const int h_upper = h_floor + 1;
        const int w_upper = w_floor + 1;
        float delta_y = (inh - h_floor);
        float delta_x = (w_upper - inw);
        float right_up_area = delta_x*delta_y ;
        return right_up_area;
    }else if(weight_kind == 2) //left_bottom
    {
        const int h_floor = floor(inh);
        const int w_floor = floor(inw);
        const int h_upper = h_floor + 1;
        const int w_upper = w_floor + 1;
        float delta_y = (h_upper - inh);
        float delta_x = (inw - w_floor);
        float left_bottom_area = delta_x*delta_y ;
        return left_bottom_area;
    }else if(weight_kind == 3) //right_bottom
    {
         const int h_floor = floor(inh);
        const int w_floor = floor(inw);
        const int h_upper = h_floor + 1;
        const int w_upper = w_floor + 1;
        float delta_y = (inh - h_floor);
        float delta_x = (inw - w_floor);
        float right_bottom_area = delta_x*delta_y ;
        return right_bottom_area;
    }
}

__global__ void my_dcn_im2col_forward_cuda(const int threadTotalNum, 
                                           const float *input, const float * offset,
                                           const int batch, const int channels, const int height, const int width,
                                           const int channels_out, const int kernel_h,const int kernel_w,
                                           const int height_out, const int width_out,
                                           const int stride_h, const int stride_w, 
                                           const int pad_h, const int pad_w, 
                                           const int dilation_h, const int dilation_w,
                                           float *output)
{
    CUDA_KERNEL_LOOP(index, threadTotalNum)
    {
        //  const int threadTotalNum = batch * channels * kernel_h * kernel_w * height_out *  width_out;
        const int width_  = index % width_out;
        const int height_ = (index / width_out) % height_out;
        const int kernel_w_ = (index / width_out / height_out ) % kernel_w;
        const int kernel_h_ = (index / width_out / height_out / kernel_w ) % kernel_h;
        const int channel_ = (index / width_out / height_out / kernel_h / kernel_w) % channels;
        const int batch_ = (index / width_out / height_out / kernel_h / kernel_w / channels) % batch;
        
        float* output_ = output + (batch_* channels + channel_) * kernel_h * kernel_w * height_out *  width_out +  height_ * width_out + width_ ;
        
        // for (int i = 0; i < kernel_h; ++i){
        //     for (int j = 0; j < kernel_w; ++j){
        const int input_height = height_ * stride_h + kernel_h_ * dilation_h - pad_h;
        const int input_width = width_ * stride_w +  kernel_w_ * dilation_w - pad_w;
        if ( 0 < input_height  && input_height < height && 0 < input_width &&  input_width < width)
        {
            const float* offset_ = offset + batch_ * ( 2 * kernel_h * kernel_w * height_out * width_out ) + height_ * width_out + width_ ;
            float offset_h = *(offset_ + ( kernel_h_ * kernel_w + kernel_w_) );
            float offset_w = *(offset_ + ( kernel_h_ * kernel_w + kernel_w_) + 1);
            const float* input_ = input + batch_ * channels * height * width + channel_ * height * width;
            // float* input_ = input + batch_ * channel * height * width + channel_ * ( height_out * width_out ) + input_height * width_out + input_width ;
            const float inh = height_ + offset_h;
            const float inw = width_ + offset_w; 
            float st = bilinear_interpolation(inh, inw, height, width, input);
            *output_ = st;
        }
            // }
        //}
    }
}


__global__ void  my_dcn_backward_cuda_for_input(
                                const int threadTotalNum,
                                const float * grad_back,
                                const float * input,
                                const float * offset,
                                const float * weight,
                                const int batch, const int channels, const int height, const int width,
                                const int channels_out, const int kernel_h, const int kernel_w,
                                const int height_out, const int width_out, const int stride_h, const int stride_w, 
                                const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
                                float * input_grad)
{
    // threadTotalNum = batch * channels * kernel_h * kernel_w * height_out *  width_out;
    // grad_back = (batch, channels, kernel_h * kernel_w, height_out * width_out)
    // input_grad = (batch, channels, height, width)
    // offset = (batch, 2*kernel_h*kernel_w, height_out, width_out)
    CUDA_KERNEL_LOOP(index, threadTotalNum)
    {
        const int width_ = index % width_out;
        const int height_ = (index / height_out) % width_out;
        const int kernel_w_ = (index / height_out / width_out) % kernel_w;
        const int kernel_h_ = (index / height_out / width_out/ kernel_w) % kernel_h;
        const int channel_ = (index / height_out / width_out/ kernel_w / kernel_h) % channels;
        const int batch_ = (index / height_out / width_out/ kernel_w / kernel_h / channels) % batch;
        const float *grad_back_ = grad_back + (((batch_ * channels + channel_ ) * kernel_h+ kernel_h_) * kernel_w + kernel_w_) * height_out * width_out + height_ * width_out + width_;
        const float *offset_h = offset + ((batch_ * channels + channel_ )* 2 * kernel_h * kernel_w + 2*kernel_h_ * kernel_w_)* height_out * width_out + height_ * width_out + width_;
        const float *offset_w = offset + ((batch_ * channels + channel_ )* 2 * kernel_h * kernel_w + 2*kernel_h_ * kernel_w_+1)* height_out * width_out + height_ * width_out + width_;
        float *input_grad_ = input_grad + (batch_ * channels + channel_)*height* width;

        const int width_input = width_ * stride_w + kernel_w_ * (dilation_w-1) + 1;
        const int height_input = height_ * stride_h + kernel_h_ * (dilation_h-1) + 1;
        const float width_plus_offset = width_input + *offset_w;
        const float height_plus_offset = height_input + *offset_h;

        const int h_floor = floor(height_plus_offset);
        const int w_floor = floor(width_plus_offset);
        const int h_upper = h_floor + 1;
        const int w_upper = w_floor + 1;
        if (h_floor>0 && h_floor<height){
            if (w_floor > 0 && w_floor < width){
                float *input_left_up = (input_grad_ +  h_floor * width + w_floor);
                float st = (*(grad_back_))*input_grad_weight(height_plus_offset, width_plus_offset, 0);
                atomicAdd(input_left_up, st);
            }
        }

        if (h_floor>0 && h_floor<height){
            if (w_upper > 0 && w_upper < width){
                float *input_right_up = (input_grad_ +  h_floor * width + w_upper);
                float st = (*(grad_back_))*input_grad_weight(height_plus_offset, width_plus_offset, 1);
                atomicAdd(input_right_up, st);
            }
        }

        if (h_upper>0 && h_upper<height){
            if (w_floor > 0 && w_floor < width){
                float *input_left_bottom = (input_grad_ +  h_upper * width + w_floor);
                float st = (*(grad_back_))*input_grad_weight(height_plus_offset, width_plus_offset, 2);
                atomicAdd(input_left_bottom, st);
            }
        }

        if (h_upper>0 && h_upper<height){
            if (w_upper > 0 && w_upper < width){
                float *input_right_bottom = (input_grad_ +  h_upper * width + w_upper);
                float st = (*(grad_back_))*input_grad_weight(height_plus_offset, width_plus_offset, 3);
                atomicAdd(input_right_bottom, st);
            }
        }
    }
}


__global__ void  my_dcn_backward_cuda_for_offset(
                                const int threadTotalNum,
                                const float * grad_back,
                                const float * input,
                                const float * offset,
                                const float * weight,
                                const int batch, const int channels, const int height, const int width,
                                const int channels_out, const int kernel_h, const int kernel_w,
                                const int height_out, const int width_out, const int stride_h, const int stride_w, 
                                const int pad_h, const int pad_w, const int dilation_h, const int dilation_w,
                                float * offset_grad)
{
    // threadTotalNum = batch * channels * kernel_h * kernel_w * height_out *  width_out;
    // grad_back = (batch, channels, kernel_h * kernel_w, height_out * width_out)
    // input_grad = (batch, channels, height, width)
    // offset = (batch, 2*kernel_h*kernel_w, height_out, width_out)
    CUDA_KERNEL_LOOP(index, threadTotalNum)
    {
        const int width_ = index % width_out;
        const int height_ = (index / height_out) % width_out;
        const int kernel_w_ = (index / height_out / width_out) % kernel_w;
        const int kernel_h_ = (index / height_out / width_out/ kernel_w) % kernel_h;
        const int channel_ = (index / height_out / width_out/ kernel_w / kernel_h) % channels;
        const int batch_ = (index / height_out / width_out/ kernel_w / kernel_h / channels) % batch;
        const float *grad_back_ = grad_back + (((batch_ * channels + channel_ ) * kernel_h+ kernel_h_) * kernel_w + kernel_w_) * height_out * width_out + height_ * width_out + width_;
        const float *offset_h = offset + ((batch_ * channels + channel_ )* 2 * kernel_h * kernel_w + 2*kernel_h_ * kernel_w_)* height_out * width_out + height_ * width_out + width_;
        const float *offset_w = offset + ((batch_ * channels + channel_ )* 2 * kernel_h * kernel_w + 2*kernel_h_ * kernel_w_+1)* height_out * width_out + height_ * width_out + width_;
        
        float *offset_grad_h = offset_grad + ((batch_ * channels + channel_ )* 2 * kernel_h * kernel_w + 2*kernel_h_ * kernel_w_)* height_out * width_out + height_ * width_out + width_;
        float *offset_grad_w = offset_grad + ((batch_ * channels + channel_ )* 2 * kernel_h * kernel_w + 2*kernel_h_ * kernel_w_ + 1)* height_out * width_out + height_ * width_out + width_;

        const int width_input = width_ * stride_w + kernel_w_ * (dilation_w-1) + 1;
        const int height_input = height_ * stride_h + kernel_h_ * (dilation_h-1) + 1;
        const float width_plus_offset = width_input + *offset_w;
        const float height_plus_offset = height_input + *offset_h;

        const float * input_ = input + (batch_ * channels + channel_ ) * height * width;

        float offset_height_weight =  grad_offset_weight(height_plus_offset, width_plus_offset, height, width, input_, 0);
        float offset_width_weight =  grad_offset_weight(height_plus_offset, width_plus_offset, height, width, input_, 1);
        *(offset_grad_h) = offset_height_weight * (*grad_back_);
        *(offset_grad_w) = offset_width_weight * (*grad_back_);
    }
}

// input = (batch, channel, height, width)
// offset = (batch, 2*kernel_h*kernel_w, height, width)
void my_dcn_im2col_forward(cudaStream_t stream,
                            const float *input,
                            const float *offset,
                            const int batch, const int channels, const int height, const int width,
                            const int channels_out, const int kernel_h,const int kernel_w,
                            const int height_out, const int width_out, 
                            const int stride_h, const int stride_w, 
                            const int pad_h, const int pad_w, 
                            const int dilation_h, const int dilation_w,
                            float *output)
{
    const int threadTotalNum = batch * channels * kernel_h * kernel_w * height_out *  width_out;
    const int blockNum =  (threadTotalNum + align_threads-1) / align_threads + 1;

    my_dcn_im2col_forward_cuda<<<blockNum, align_threads, 0, stream>>>(threadTotalNum, input, offset, 
                                                                      batch, channels, height, width,
                                                                      channels_out, kernel_h, kernel_w,
                                                                      height_out, width_out, 
                                                                      stride_h, stride_w, 
                                                                      pad_h, pad_w, 
                                                                      dilation_h, dilation_w,
                                                                      output);
}


void my_dcn_backward_for_input(cudaStream_t stream,
                               const float * grad_back,
                               const float * input,
                               const float * offset,
                               const float * weight,
                               const int batch, const int channels, const int height, const int width,
                               const int channels_out,const int kernel_h,const int kernel_w,
                               const int height_out,const int width_out,const int stride_h,const int stride_w, 
                               const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
                               float * input_grad)
{
    const int threadTotalNum = batch * channels * kernel_h * kernel_w * height_out *  width_out;
    const int blockNum = (threadTotalNum + align_threads-1) / align_threads + 1;
    my_dcn_backward_cuda_for_input<<<blockNum, align_threads, 0, stream>>>
                                (
                                threadTotalNum,
                                grad_back,
                                input,
                                offset,
                                weight,
                                batch, channels,  height,  width,
                                channels_out, kernel_h, kernel_w,
                                height_out, width_out, stride_h, stride_w, 
                                pad_h, pad_w, dilation_h, dilation_w,
                                input_grad);
}


void my_dcn_backward_for_offset(cudaStream_t stream,
                               const float * grad_back,
                               const float * input,
                               const float * offset,
                               const float * weight,
                               const int batch, const int channels, const int height, const int width,
                               const int channels_out,const int kernel_h,const int kernel_w,
                               const int height_out,const int width_out,const int stride_h,const int stride_w, 
                               const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
                               float * offset_grad)
{
    const int threadTotalNum = batch * channels * 2*kernel_h * kernel_w * height_out *  width_out;
    const int blockNum = (threadTotalNum + align_threads-1) / align_threads + 1;
    my_dcn_backward_cuda_for_offset<<<blockNum, align_threads, 0, stream>>>
                                (
                                threadTotalNum,
                                grad_back,
                                input,
                                offset,
                                weight,
                                batch, channels,  height,  width,
                                channels_out, kernel_h, kernel_w,
                                height_out, width_out, stride_h, stride_w, 
                                pad_h, pad_w, dilation_h, dilation_w,
                                offset_grad);
}
