# Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
### K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure, IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018. DOI: [10.1109/ICIP.2018.8451020](https://doi.org/10.1109/ICIP.2018.8451020)

## Update
add TensorFlow implentation and pretrained model.  
- Dependencies: 
    - Python 3
    - TensorFlow 1.XX (1.10 or newer)
    - NumPy
    - Pillow
    - NVIDIA GPU + CUDA (if running in GPU mode)

- Dataset:
    - You need to download the testing datasets to run the demo test for different tasks. We summarize the datasets [here](https://tumde-my.sharepoint.com/:u:/g/personal/kai_cui_tum_de/EYL2pOxmLylDsRgcO6U7fIAB91lH3cpqjGgVScd1QKIfZA?e=Pm41ch). Unzip the datasets and put them into the data folder. If you have your own dataset, please follow the [readme](./TensorFlow_reimplmentation/data/readme.md) in the data folder to organize the dataset.
- Usage:
    - run `python main_py3_tfrecord.py` to test the Kodak dataset.  
    - When testing other datasets, simply add ` --test_set NAME`, e.g., `python main_py3_tfrecord.py --test_set McM`  
    - It also supports the ensemble testing mode, run `python main_py3_tfrecord.py --phase ensemble`

## Original MatConvNet Implementation
1. Please download the matconvnet toolbox from http://www.vlfeat.org/matconvnet/ and install it according to the instructions from their website.
2. Please go to the folder ./MatConvnet_implementation
3. Please copy the ./MatConvnet_implementation folder into the following path, ./Matconvnet-1.0-beta2X/examples/
4. Please copy the customized layer functions vl_nnsplit.m, vl_nnsplit_new.m in the ./customized_layers/ to ./Matconvnet-1.0-beta2X/matlab/; Copy the Split.m, Split_new.m in the ./customized_layers/ to ./Matconvnet-1.0-beta2X/matlab/+dagnn/;
5. The script test_CDMNet.m is a demo for testing using the trained model which is stored in ./model/CNNCDM.mat
6. In case you would like to train the network. The traing dataset used is the Waterloo Exploration Database. Please download the dataset here https://ece.uwaterloo.ca/~k29ma/exploration/ and put all the images in ./pristine_images/; and then run mosaicked_image_generation to generate the bilinear initial CDM input of the network; and then run train_CDMNet_MSE for training.
7. Please read our paper for more details!
8. Have fun!

```
@INPROCEEDINGS{LMT2018-1279,  
 author = {Kai Cui AND Zhi Jin AND Eckehard Steinbach},
 title = {Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure},
 booktitle = {{IEEE} International Conference on Image Processing ({ICIP} 2018)},
 month = {Oct},
 year = {2018},
 address = {Athens, Greece}
}
```
---
## Maintainer:

[@Kai Cui](https://github.com/amnesiack) (<kai.cui@tum.de>)  
Lehrstuhl fuer Medientechnik ([LMT](https://www.ei.tum.de/en/lmt/home/))  
Technische Universitaet Muenchen ([TUM](https://www.tum.de))  
Last modified 06.02.2021

---

## License
[![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)  
This project is released under the Apache 2.0 license.