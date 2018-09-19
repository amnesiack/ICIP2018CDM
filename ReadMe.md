## Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
### K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
#### Kai Cui <Kai.cui@tum.de>
#### Lehrstuhl fuer Medientechnik
#### Technische Universitaet Muenchen
#### Last modified 17.05.2018

1. Please download the matconvnet toolbox from http://www.vlfeat.org/matconvnet/ and install it according to the instructions from their website.
2. Please uncompress the *.zip file and put the whole project fold in the following path, ./Matconvnet-1.0-beta2X/examples/
3. Please copy the customized layer functions vl_nnsplit.m, vl_nnsplit_new.m in the ./customized_layers/ to ./Matconvnet-1.0-beta2X/matlab/; Copy the Split.m, Split_new.m in the ./customized_layers/ to ./Matconvnet-1.0-beta2X/matlab/+dagnn/;
4. The script test_CDMNet.m is a demo for testing using the trained model which is stored in ./model/CNNCDM.mat
5. In case you would like to train the network. The traing dataset used is the Waterloo Exploration Database. Please download the dataset here https://ece.uwaterloo.ca/~k29ma/exploration/ and put all the images in ./pristine_images/; and then run mosaicked_image_generation to generate the bilinear initial CDM input of the network; and then run train_CDMNet_MSE for training.
6. Please read our paper for more details!
7. Have fun!


@INPROCEEDINGS{LMT2018-1279,
author = {Kai Cui AND Zhi Jin AND Eckehard Steinbach},
title = {Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure},
booktitle = {{IEEE} International Conference on Image Processing ({ICIP} 2018)},
month = {Oct},
year = {2018},
address = {Athens, Greece}
}
