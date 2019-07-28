# OctaveConv_Caffe
This repository contains a [Caffe](https://github.com/BVLC/caffe) implementation of the paper [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049).  
## OctaveUpsample
Provide octaveupsample layer to support octave convlution.  
### Forward:    
 <img src="https://github.com/chaipangpang/OctaveConv_Caffe/blob/master/pics/forward.jpg" width="447" height="220" alt="forward"/>  
### Backward:  
  <img src="https://github.com/chaipangpang/OctaveConv_Caffe/blob/master/pics/backward.jpg" width="447" height="200" alt="backward"/>  
## Example 
Here is a performance on the scene classification task of resnet18 in [AI Challenger](https://challenger.ai/?lan=zh)
<img src="https://github.com/chaipangpang/OctaveConv_Caffe/blob/master/pics/acc.jpg" width="612" height="335" alt="top-3 acc"/>  
## Usage  
### Prerequisites  
   Caffe  
   CUDA8.0  
   cudnn5.0  

