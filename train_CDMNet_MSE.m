%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018
function train_CDMNet_MSE(varargin)
clc
% -------------------------------------------------------------------------
% Part 1: prepare the data
% -------------------------------------------------------------------------
run(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'matlab', 'vl_setupnn.m'));
imdb = getCDMNetData;
imdb_bilinear = getCDMNetData_bilinear;
% -------------------------------------------------------------------------
% Part 2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = CDMNet;
% -------------------------------------------------------------------------
% Part 3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.batchSize    = 64;
opts.train.continue     = true;
opts.train.gpus         = [1];
opts.train.prefetch     = false;
opts.train.expDir       = './model/';
opts.train.learningRate = [0.001*ones(1, 20), 5e-4*ones(1,20), 1e-4*ones(1, 20), 5e-5*ones(1, 20)];
opts.train.weightDecay  = 0.0005;
opts.train.numEpochs    = numel(opts.train.learningRate);
opts.train.derOutputs   = {'LossAll', 1};
[opts, ~]               = vl_argparse(opts.train, varargin);
if(~isdir(opts.expDir)), mkdir(opts.expDir); end

% Call training function in MatConvNet
[net_fc, info_fc] = cnn_train_dagCDMNet(net, imdb, @getBatch, opts);
% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch, opts)
% get img from the imdb
img     = imdb.images.data(:, :, :, batch) ;
label   = img;
input   = gpuArray(single(imdb_bilinear.images.data(:, :, :, batch)));
inputs  = {'input',(input),'label',(label)};
end
% --------------------------------------------------------------------

end