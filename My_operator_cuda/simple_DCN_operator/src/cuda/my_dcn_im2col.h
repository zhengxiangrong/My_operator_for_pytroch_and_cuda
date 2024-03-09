#ifndef MY_DCN_IM2COL_H
#define MY_DCN_IM2COL_H

#ifdef __cplusplus
extern "C"
{
#endif


void my_dcn_im2col_forward(cudaStream_t stream,
                            const float *input,
                            const float *offset,
                            const int batch, const int channels, const int height, const int width,
                            const int channels_out, const int kernel_h,const int kernel_w,
                            const int height_out, const int width_out,
                            const int stride_h, const int stride_w,
                            const int pad_h, const int pad_w,
                            const int dilation_h, const int dilation_w,
                            float *output);

void my_dcn_backward_for_input(cudaStream_t stream,
                               const float * grad_back,
                               const float * input,
                               const float * offset,
                               const float * weight,
                               const int batch, const int channels, const int height, const int width,
                               const int channels_out,const int kernel_h,const int kernel_w,
                               const int height_out,const int width_out,const int stride_h,const int stride_w, 
                               const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
                               float * input_grad);

void my_dcn_backward_for_offset(cudaStream_t stream,
                               const float * grad_back,
                               const float * input,
                               const float * offset,
                               const float * weight,
                               const int batch, const int channels, const int height, const int width,
                               const int channels_out,const int kernel_h,const int kernel_w,
                               const int height_out,const int width_out,const int stride_h,const int stride_w, 
                               const int pad_h,const int pad_w,const int dilation_h,const int dilation_w,
                               float * offset_grad);
#ifdef __cplusplus
}
#endif
#endif