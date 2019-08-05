# OctaveConv_Caffe
This repository contains a [Caffe](https://github.com/BVLC/caffe) implementation of the paper [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049).  
## OctaveUpsample
Provide [octaveupsample layer](https://github.com/chaipangpang/OctaveConv_Caffe/tree/master/octave_upsample_layer) to support octave convolution.  
Forward:    
 <img src="https://github.com/chaipangpang/OctaveConv_Caffe/blob/master/pics/forward.jpg" width="447" height="220" alt="forward"/>    
Backward:   
<img src="https://github.com/chaipangpang/OctaveConv_Caffe/blob/master/pics/backward.jpg" width="447" height="200" alt="backward"/>  
## Example  
Here is a performance on the scene classification task of resnet18 in [AI Challenger](https://challenger.ai/?lan=zh).The experiment in GTX1080Ti*2, batchsize=128, did not use any data augmentation without mirror, and did not load the pre-training model.  
More other models are coming soon.  
<img src="https://github.com/chaipangpang/OctaveConv_Caffe/blob/master/pics/acc.jpg" width="612" height="335" alt="top-3 acc"/>  
## Usage  
### Prerequisites  
[Caffe](https://github.com/BVLC/caffe)    
[CUDA8.0](https://developer.nvidia.com/cuda-toolkit)  
[cudnn5.0](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)     
### How to build 
Modify [caffe.proto](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto) like this：  
Add option in LayerParameter.
```
message LayerParameter {
optional OctaveUpsampleParameter octaveupsample_param = Your last ID;
}
```  
Add message at the end.
```
message OctaveUpsampleParameter {
  // DEPRECATED. No need to specify upsampling scale factors when
  // exact output shape is given by upsample_h, upsample_w parameters.
  optional uint32 scale = 1 [default = 2];
  // DEPRECATED. No need to specify upsampling scale factors when
  // exact output shape is given by upsample_h, upsample_w parameters.
  optional uint32 scale_h = 2;
  // DEPRECATED. No need to specify upsampling scale factors when
  // exact output shape is given by upsample_h, upsample_w parameters.
  optional uint32 scale_w = 3;
  // DEPRECATED. Specify exact output height using upsample_h. This
  // parameter only works when scale is 2
  optional bool pad_out_h = 4 [default = false];
  // DEPRECATED. Specify exact output width using upsample_w. This
  // parameter only works when scale is 2
  optional bool pad_out_w = 5 [default = false];
  optional uint32 upsample_h = 6;
  optional uint32 upsample_w = 7;
}
```
### How to use  
Reference OctaveUpsample layer in [resnet18_octave_0.5_train.prototxt](https://github.com/chaipangpang/OctaveConv_Caffe/blob/master/model_example/Resnet18/resnet18_octave_0.5_train.prototxt).  

## Reference  
1.Caffe:[pooling layer](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/pooling_layer.cpp)  
2.caffe-segnet-cudnn5:[upsample layer](https://github.com/TimoSaemann/caffe-segnet-cudnn5/blob/master/src/caffe/layers/upsample_layer.cpp)   
3.[OctaveConv_pytorch](https://github.com/lxtGH/OctaveConv_pytorch)  
4.[OctaveConv](https://github.com/terrychenism/OctaveConv)  

