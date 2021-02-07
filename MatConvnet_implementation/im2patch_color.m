%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018
function patch = im2patch_color(image, patchsize)

[a, b, ~] = size(image);
a1 = floor(a/patchsize)*patchsize;
b1 = floor(b/patchsize)*patchsize;

image_crop = image(1:a1, 1:b1, :);
count = 1;
for i = 1:patchsize:a1
    for j = 1:patchsize:b1
        patch(:,:,:,count) = image_crop(i:i+patchsize-1,j:j+patchsize-1,:);
        count = count+1;
    end
end

end