%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018

function [r, g] = vl_nnsplit_new(X, dr)
% SplitLayer
% X	: w x h x c x b
% r : w x h x 1 x b
% g : w x h x 1 x b
% b : w x h x 1 x b

if nargin<2
    %forward
    r(:,:,1,:) = X(:,:,1,:);
    g(:,:,1,:) = X(:,:,2,:);
else
    %backward
    [w, h, ~, batch] = size(dr);
    r = gpuArray(zeros([w, h, 2, batch],'single'));
    for i = 1 : batch
        temp = cat(3,X(:,:,:,i), dr(:,:,:,i));
        r(:,:,:,i) = temp;
    end
end

end