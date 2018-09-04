%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Chair of Media Technology
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018

% Generating the bilinear initialization
clc
clear
Filepath = '.\pristine_images\';
DSTpath  = '.\pristine_images_mosaick\';
fileAll   = dir(Filepath);
fileAll   = fileAll(~[fileAll.isdir]);
for i = 1:length(fileAll)
    ori = imread([Filepath fileAll(i).name]);
    [m, n, l]  = size(ori);
    ori        = ori(1:floor(m/2)*2, 1:floor(n/2)*2,:);
    
    ori_mosaick = demosaicing(ori);
    
    if mod(m,2)% dealing with the odd hight or width
        ori_mosaick = [ori_mosaick;ori_mosaick(end,:,:)];
    end
    if mod(n,2)
        ori_mosaick = [ori_mosaick,ori_mosaick(:,end,:)];
    end
    
    imwrite(uint8(ori_mosaick), [DSTpath fileAll(i).name])
end
