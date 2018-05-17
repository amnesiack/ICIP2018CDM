## Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure
                                  ICIP 2018
                     Kai Cui, Zhi Jin, Eckehard Steinbach
             Chair of Media Technology, Technical University of Munich

### Abstract
Color demosaicking (CDM) is a critical first step for the acquisition of high-quality RGB images with single chip cameras. Conventional CDM approaches are mostly based on interpolation schemes and hand-crafted image priors, which result in unpleasant visual artifacts in some cases. Motivated by the special characteristics of inter-channel correlations (higher correlations for R/G and G/B channels than that for R/B), in this paper, a 3-stage convolutional neural network (CNN) structure for CDM is proposed. In the first stage,  the G channel is reconstructed independently. Then, by using the reconstructed G channel as guidance, the R and B channels are recovered in the second stage. Finally, high-quality RGB color images are reconstructed in the third stage. The objective and visual quality evaluation results show that the proposed structure achieves noticeable quality improvements in comparison to the state-of-the-art approaches.

### Proposed Structure
Inspired by inter-channel correlation, in this paper, we propose a 3-stage CNN structure for color demosaicking.
![CNNCDM-3Stage](图片链接 "optional title")
### Evaluation

### Reference

### Source Code
