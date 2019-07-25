#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/octave_upsample_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void OctaveupsampleForward(const int nthreads, const Dtype* const bottom_data,
	const int num, const int channels,
	const int bottom_height, const int bottom_width,
	const int upsample_height, const int upsample_width,
	const int copy_height, const int copy_width,
	Dtype* const top_data)
  {
    CUDA_KERNEL_LOOP(index, nthreads) {
	const int bw = index % bottom_width;
	const int bh = (index / bottom_width) % bottom_height;
	const int c = (index / bottom_width / bottom_height) % channels;
	const int n = index / bottom_width / bottom_height / channels;
	 //bottom index = index
	 int idx_b = index;
	 //caulculate bottom_slice
	 //const Dtype* const bottom_slice =
	 //    bottom_data + (n * channels + c) * bottom_height * bottom_width;
	 //caulculate top_slice
	  //Dtype*  top_slice = 
		 //top_data + (n * channels + c) * upsample_height * upsample_width;
	 for (int ch = 0; ch < copy_height; ++ch) {
		 for (int cw = 0; cw < copy_width; ++cw) {
			 //caulculate index in top
			 int idx_t =  (n * channels + c) * upsample_height * upsample_width +
				              bh*upsample_width*copy_height + bw* copy_width + ch*upsample_width + cw;
			 top_data[idx_t] = bottom_data[idx_b];
		 }
	  }
    }
  }

template <typename Dtype>
void OctaveUpsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  int bottom_count = bottom[0]->count();
  OctaveupsampleForward<Dtype> << <CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS >> >(
	  bottom_count, bottom_data, bottom[0]->num(), channels_,
	  height_, width_,
	  upsample_h_, upsample_w_,
      copy_w, copy_h
	  , top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void OctaveupsampleBackward(const int nthreads, const Dtype* const  top_diff,
	const int num, const int channels,
	const int upsample_height, const int upsample_width,
	const int copy_height, const int copy_width,
	const int bottom_height, const int bottom_width,
	Dtype* const bottom_diff)
{
    CUDA_KERNEL_LOOP(index, nthreads) {
		const int bw = index % bottom_width;
		const int bh = (index / bottom_width) % bottom_height;
		const int c = (index / bottom_width / bottom_height) % channels;
		const int n = index / bottom_width / bottom_height / channels;
		//bottom index = index
		int idx_b = index;
		//caulculate bottom_slice
		 //Dtype*  bottom_slice =
			//bottom_diff + (n * channels + c) * bottom_height * bottom_width;
		//caulculate top_slice
		//const Dtype* const top_slice =
		//	top_diff + (n * channels + c) * upsample_height * upsample_width;
		for (int ch = 0; ch < copy_height; ++ch) {
			for (int cw = 0; cw < copy_width; ++cw) {
				//caulculate index in top
				int idx_t = (n * channels + c) * upsample_height * upsample_width  +
					             bh*upsample_width*copy_height + bw* copy_width + ch*upsample_width + cw;
				bottom_diff[idx_b] += top_diff[idx_t];
			}
		}
    }
  }

template <typename Dtype>
void OctaveUpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);
	OctaveupsampleBackward<Dtype> << <CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS >> >(
		bottom_count, top_diff, bottom[0]->num(), channels_,
		upsample_h_, upsample_w_,
		copy_w, copy_h,
		height_, width_,
		bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(OctaveUpsampleLayer);
}  // namespace caffe
