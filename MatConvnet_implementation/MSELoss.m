%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018
classdef MSELoss < dagnn.Loss
    methods
        function outputs = forward(obj, inputs, params)
            [w,h,d,~]       = size(inputs{1});
            t               = bsxfun(@minus,inputs{2},inputs{1});
            t               = gather(t);
            t               = reshape(t, 1, []);
            outputs{1}      = sum(t.^2)/(w*h);
            n               = obj.numAveraged ;
            m               = n + size(inputs{1},4) ;
            obj.average     = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            [w,h,d,b]    = size(inputs{2});
            Y            = gather(bsxfun(@minus,inputs{1},inputs{2}));
            derInputs{1} = bsxfun(@times, derOutputs{1}, Y);
            derInputs{2} = [] ;
            derParams    = {} ;
        end
        
        function obj = MSELoss(varargin)
            obj.load(varargin) ;
        end
        
    end
end