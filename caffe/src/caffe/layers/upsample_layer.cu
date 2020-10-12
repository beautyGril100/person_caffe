#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/******************************added by hwzhu start****************************/

__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

__device__ int translate_idx_inv(
    int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*scale_factor+off_x;
  z = z*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

template <typename Dtype>
__global__ void upscale(const Dtype *input, Dtype *output,
        int no_elements, int scale_factor, int d1, int d2, int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
  output[ii]=input[ipidx];
}

template <typename Dtype>
__global__ void downscale(Dtype *gradInput_data, const Dtype *gradOutput_data,
                          int no_elements, int scale_factor, int d1, int d2,
                          int d3) {
  int ii = threadIdx.x + blockDim.x * blockIdx.x;
  if (ii >= no_elements) return;
  for (int i = 0; i < scale_factor; i++) {
    for (int j = 0; j < scale_factor; j++) {
      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
      gradInput_data[ii] += gradOutput_data[ipidx];
    }
  }
}


/******************************added by hwzhu end  ****************************/

template <typename Dtype>
  __global__ void UpsampleForward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, const Dtype* bottom_data,
      const Dtype* bottom_mask, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int offset = index / (in_w * in_h) * out_w * out_h;
      int upsample_idx = static_cast<int>(bottom_mask[index]);
      top_data[offset + upsample_idx] = bottom_data[index];
    }
  }

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /******************************added by hwzhu start********************************/
  if (only_one_bottom_blob_flag){
    int d1, d2, d3;

    d1 = top[0]->shape(1);
    d2 = top[0]->shape(2);
    d3 = top[0]->shape(3);
  
    int no_elements = top[0]->count();
  
    upscale<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(no_elements), CAFFE_CUDA_NUM_THREADS>>>(
        bottom[0]->gpu_data(),
        top[0]->mutable_gpu_data(), no_elements, scale_, d1, d2, d3);
        
    return;
  }
  /******************************added by hwzhu end  ********************************/
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_mask = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  int bottom_count = bottom[0]->count();
  UpsampleForward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_count, bottom[0]->width(), bottom[0]->height(), 
      top[0]->width(), top[0]->height(), bottom_data, bottom_mask, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
  __global__ void UpsampleBackward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, const Dtype* top_diff,
      const Dtype* bottom_mask, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int offset = index / (in_w * in_h) * out_w * out_h;
      int upsample_idx = static_cast<int>(bottom_mask[index]);
      bottom_diff[index] = top_diff[offset + upsample_idx];
    }
  }

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  /***************************************added by hwzhu start*****************************************/
  if (only_one_bottom_blob_flag){
    if (propagate_down[0]){
      int d1, d2, d3;
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      d1 = bottom[0]->shape(1);
      d2 = bottom[0]->shape(2);
      d3 = bottom[0]->shape(3);
      int no_elements = bottom[0]->count();
      caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
      downscale<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(no_elements), CAFFE_CUDA_NUM_THREADS>>>(
          bottom_diff, top[0]->gpu_diff(), no_elements, scale_, d1, d2, d3);
    } 
    return;
  }
  /***************************************added by hwzhu end  *****************************************/
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_mask = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);
    UpsampleBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, bottom[0]->width(), bottom[0]->height(), 
        top[0]->width(), top[0]->height(), top_diff, bottom_mask, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(UpsampleLayer);


}  // namespace caffe
