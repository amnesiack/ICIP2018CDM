## <center> Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure </center>
<center> ICIP 2018 </center>
<center> Kai Cui, Zhi Jin, Eckehard Steinbach </center>
<center> Chair of Media Technology, Technical University of Munich </center>

### Abstract
Color demosaicking (CDM) is a critical first step for the acquisition of high-quality RGB images with single chip cameras. Conventional CDM approaches are mostly based on interpolation schemes and hand-crafted image priors, which result in unpleasant visual artifacts in some cases. Motivated by the special characteristics of inter-channel correlations (higher correlations for R/G and G/B channels than that for R/B), in this paper, a 3-stage convolutional neural network (CNN) structure for CDM is proposed. In the first stage,  the G channel is reconstructed independently. Then, by using the reconstructed G channel as guidance, the R and B channels are recovered in the second stage. Finally, high-quality RGB color images are reconstructed in the third stage. The objective and visual quality evaluation results show that the proposed structure achieves noticeable quality improvements in comparison to the state-of-the-art approaches.

### Proposed Structure
Inspired by inter-channel correlation, in this paper, we propose a 3-stage CNN structure for color demosaicking.

![CNNCDM-3Stage](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/CDM_new3stages1.4_compact.png "Structure of the proposed 3-stage CNN scheme")

### Evaluation

kodim19.png Original Image
  
![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/Origin_Image_mark_kodim19.png)
  
kodim19.png AHD (38.29dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/AHD.png)

kodim19.png DLMMSE (41.05dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/DLMMSE.png)

kodim19.png DLMMSE (41.83dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/GBTF.png)

kodim19.png LDI-NAT (37.78dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/LDI-NAT.png)

kodim19.png RI (39.16dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/RI.png)

kodim19.png MLRI (39.92dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/MLRI.png)

kodim19.png ARI (40.50dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/ARI.png)

kodim19.png RI-modified (39.46dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/RI-modified.png)

kodim19.png ARI-modified (40.57dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/ARI-modified.png)

kodim19.png 2-stage (41.74dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/2-Stage.png)

kodim19.png Proposed (42.82dB)

![](https://github.com/amnesiack/ICIP2018CDM/raw/master/docs/Recon/Proposed.png)

### Source Code
https://github.com/amnesiack/ICIP2018CDM
