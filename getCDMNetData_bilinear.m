%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018
% --------------------------------------------------------------------
function CSNet_Data = getCDMNetData_bilinear
% --------------------------------------------------------------------
[patchT, patchV] = TrainData;
patchT_all = patchT;
patchV_all = patchV;
patchT_all = patchT_all(:,:,:,1:floor(size(patchT_all,4)/128)*128);
patchV_all = patchV_all(:,:,:,1:floor(size(patchV_all,4)/128)*128);
set        = [ones(1,size(patchT_all, 4)) 2*ones(1,size(patchV_all, 4))];
data       = cat(4, patchT_all, patchV_all);

CSNet_Data.images.data      = data ;
CSNet_Data.images.set       = set ;
CSNet_Data.meta.sets        = {'train', 'val'} ;
end

function [patchTrain, patchVal] = TrainData
blocksize = 50;
%% Training Dataset
Filepath = '.\pristine_images_mosaick\';
fileAll   = dir(Filepath);
fileAll   = fileAll(~[fileAll.isdir]);
RanIdx    = randperm(length(fileAll));
fileAll   = fileAll(RanIdx);
fileTrain = fileAll(1:end-100, :);

imagesTrain = cell(numel(fileTrain),1);
patchTrain  = uint8([]);
% Color Transform for all frames
for i = 1 : numel(fileTrain)
    fprintf('The %dth training image\n', i);
    imagesTrain{i} = imread([Filepath fileTrain(i).name]);
    patch_temp     = im2patch_color(imagesTrain{i}, blocksize);
    patchTrain     = cat(4, patchTrain, patch_temp);
end
%% Validaing Dataset
fileVal = fileAll(end-99:end, :);
imagesVal = cell(numel(fileVal),1);
patchVal  = uint8([]);
% Color Transform for all frames
for i = 1 : numel(fileVal)
    fprintf('The %dth validation image\n', i);
    imagesVal{i} = imread([Filepath fileVal(i).name]);
    patch_temp   = im2patch_color(imagesTrain{i}, blocksize);
    patchVal     = cat(4, patchVal, patch_temp);
end
end
