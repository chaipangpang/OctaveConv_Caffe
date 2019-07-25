#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/octave_upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void OctaveUpsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	OctaveUpsampleParameter octaveupsample_param = this->layer_param_.octaveupsample_param();
	CHECK((octaveupsample_param.has_upsample_h() && octaveupsample_param.has_upsample_w())
		|| (!octaveupsample_param.has_scale() && octaveupsample_param.has_scale_h()
		&& octaveupsample_param.has_scale_w())
		|| (!octaveupsample_param.has_scale_h() && !octaveupsample_param.has_scale_w()))
      << "upsample_h & upsample_w are required, else (DEPRECATED) "
      << "scale OR scale_h & scale_w are required.";

	if (octaveupsample_param.has_upsample_h() && octaveupsample_param.has_upsample_w()) {
		upsample_h_ = octaveupsample_param.upsample_h();
		upsample_w_ = octaveupsample_param.upsample_w();
    CHECK_GT(upsample_h_, 1);
    CHECK_GT(upsample_w_, 1);
  } else {
    LOG(INFO) << "Params 'pad_out_{}_' are deprecated. Please declare upsample"
        << " height and width useing the upsample_h, upsample_w parameters.";
	if (!octaveupsample_param.has_scale_h()) {
		scale_h_ = scale_w_ = octaveupsample_param.scale();
      CHECK_GT(scale_h_, 1);
    } else {
		scale_h_ = octaveupsample_param.scale_h();
		scale_w_ = octaveupsample_param.scale_w();
      CHECK_GT(scale_h_, 1);
      CHECK_GT(scale_w_, 1);
    }
	pad_out_h_ = octaveupsample_param.pad_out_h();
	pad_out_w_ = octaveupsample_param.pad_out_w();
    CHECK(!pad_out_h_ || scale_h_ == 2) 
        << "Output height padding compensation requires scale_h == 2, otherwise "
        << "the output size is ill-defined.";
    CHECK(!pad_out_w_ || scale_w_ == 2) 
        << "Output width padding compensation requires scale_w == 2, otherwise "
        << "the output size is ill-defined.";
    upsample_h_ = upsample_w_ = -1;  // flag to calculate in Reshape
  }
}

template <typename Dtype>
void OctaveUpsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, ";


  if (upsample_h_ <= 0 || upsample_w_ <= 0) {
    upsample_h_ = bottom[0]->height() * scale_h_ - int(pad_out_h_);
    upsample_w_ = bottom[0]->width() * scale_w_ - int(pad_out_w_);
  }
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), upsample_h_,
      upsample_w_);
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  //caulculate  w&h for copy bottom 
  copy_w = top[0]->width() / bottom[0]->width();
  copy_h = top[0]->height() / bottom[0]->height();

}

template <typename Dtype>
void OctaveUpsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //const Dtype* bottom_mask_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // Initialize
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		  for (int bh = 0; bh < height_; ++bh) {
			  for (int bw = 0; bw < width_; ++bw) {
				 //caulculate index in bottom
				  int idx_b = bh*bottom[0]->width() + bw;
				  for (int ch = 0; ch < copy_h; ++ch) {
					  for (int cw = 0; cw < copy_w; ++cw) {
						  //caulculate index in top
						  int idx_t = bh*top[0]->width()*copy_h + bw* copy_w + ch*top[0]->width() + cw;
						  if (idx_t >= upsample_h_ * upsample_w_) {
							  // this can happen if the pooling layer that created the input mask
							  // had an input with different size to top[0]
							  LOG(FATAL) << "upsample top index " << idx_t << " out of range - "
								  << "check scale settings match input pooling layer's "
								  << "downsample setup";
						  }
						  top_data[idx_t] = bottom_data[idx_b];
					  }
				   }
			    }
			 }
		  // compute offset
		  bottom_data += bottom[0]->offset(0, 1);
		  top_data += top[0]->offset(0, 1);
	  }
  }
}

template <typename Dtype>
void OctaveUpsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_mask_data = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    const int bottom_count = bottom[0]->count();
    caffe_set(bottom_count, Dtype(0), bottom_diff);

	// The main loop
	for (int n = 0; n < bottom[0]->num(); ++n) {
		for (int c = 0; c < channels_; ++c) {
			for (int ph = 0; ph < height_; ++ph) {
				for (int pw = 0; pw < width_; ++pw) {
					//caulculate index in bottom
					int idx_b = ph*bottom[0]->width() + pw;
					for (int h = 0; h < copy_h; ++h) {
						for (int w = 0; w < copy_w; ++w) {
							//caulculate index in top
							int idx_t = ph*top[0]->width()*copy_h + pw* copy_w + h*top[0]->width() + w;

							if (idx_t >= upsample_h_ * upsample_w_) {
								// this can happen if the pooling layer that created the input mask
								// had an input with different size to top[0]
								LOG(FATAL) << "upsample top index " << idx_t << " out of range - "
									<< "check scale settings match input pooling layer's downsample setup";
							}
							bottom_diff[idx_b] += top_diff[idx_t];
						}
					}
				}
			}
			// compute offset
			bottom_diff += bottom[0]->offset(0, 1);
			top_diff += top[0]->offset(0, 1);
		}
	}

  }
}


#ifdef CPU_ONLY
STUB_GPU(UpsampleLayer);
#endif

INSTANTIATE_CLASS(OctaveUpsampleLayer);
REGISTER_LAYER_CLASS(OctaveUpsample);

}  // namespace caffe
