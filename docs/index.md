## Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure
<center> ICIP 2018 </center>
<center> Kai Cui, Zhi Jin, Eckehard Steinbach </center>
<center> Chair of Media Technology, Technical University of Munich </center>

### Abstract
Color demosaicking (CDM) is a critical first step for the acquisition of high-quality RGB images with single chip cameras. Conventional CDM approaches are mostly based on interpolation schemes and hand-crafted image priors, which result in unpleasant visual artifacts in some cases. Motivated by the special characteristics of inter-channel correlations (higher correlations for R/G and G/B channels than that for R/B), in this paper, a 3-stage convolutional neural network (CNN) structure for CDM is proposed. In the first stage,  the G channel is reconstructed independently. Then, by using the reconstructed G channel as guidance, the R and B channels are recovered in the second stage. Finally, high-quality RGB color images are reconstructed in the third stage. The objective and visual quality evaluation results show that the proposed structure achieves noticeable quality improvements in comparison to the state-of-the-art approaches.

### Proposed Structure
Inspired by inter-channel correlation, in this paper, we propose a 3-stage CNN structure for color demosaicking.

![CNNCDM-3Stage](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/CDM_new3stages1.4_compact.png "Structure of the proposed 3-stage CNN scheme")

### Evaluation


### Source Code
https://github.com/amnesiack/ICIP2018CDM

@INPROCEEDINGS{LMT2018-1279,
author = {Kai Cui AND Zhi Jin AND Eckehard Steinbach},
title = {Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure},
booktitle = {{IEEE} International Conference on Image Processing ({ICIP} 2018)},
month = {Oct},
year = {2018},
address = {Athens, Greece}
}
