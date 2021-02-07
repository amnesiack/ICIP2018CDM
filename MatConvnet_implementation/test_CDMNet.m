%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 04.09.2018
%% Initialization
clc
clear;
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'matlab', 'vl_setupnn.m'));
%% Test image reading
img       = imread('./Test_Images/kodim01.png');
img       = single(img);
[w,h,~]   = size(img);
img       = img(1:floor(w/2)*2, 1:floor(h/2)*2, :);
patchTest = img;
patchNorm = single(demosaicing(img));
PSNR_intp = imcpsnr(img, patchNorm, 255, 10);
CSNR_intp = impsnr(img, patchNorm, 255, 10);
fprintf('Bilinear Interpolation: PSNR = %.2f dB\n', PSNR_intp)
fprintf('R: PSNR = %.2f dB, G: PSNR = %.2f dB, B: PSNR = %.2f dB\n\n', CSNR_intp(1), CSNR_intp(2), CSNR_intp(3))
%% Test using different epoches
nets_name = strcat('./model/CNNCDM.mat');
netstruct = load(nets_name);
net       = dagnn.DagNN.loadobj(netstruct.net);
net.mode  = 'test';
net.conserveMemory = true;
net.move('cpu');
tic
net.eval({'input', patchNorm});
toc
index  = net.getVarIndex('final_RGB_final');
result = gather(net.vars(index).value);

result_CDMNet = result;
result_CDMNet(result_CDMNet>255) = 255;
result_CDMNet(result_CDMNet<0)   = 0;
result_CDMNet = round(result_CDMNet);

PSNR_Current = imcpsnr(patchTest, result_CDMNet, 255, 10);
CSNR_Current = impsnr(patchTest, result_CDMNet, 255, 10);
SSIM_Current = ssim(patchTest, result_CDMNet);
FSIM_Current = FeatureSIM(patchTest, result_CDMNet);
fprintf('PSNR = %.2f dB, SSIM = %2f, FSIM = %2f\n', PSNR_Current, SSIM_Current, FSIM_Current)
fprintf('R: PSNR = %.2f dB, G: PSNR = %.2f dB, B: PSNR = %.2f dB\n\n', CSNR_Current(1), CSNR_Current(2), CSNR_Current(3))