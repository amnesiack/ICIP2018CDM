%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018
classdef Split < dagnn.ElementWise
    % Split DagNN Split layer.
    
    properties
        dim = 3;
    end
    
    methods
        
        function outputs = forward(obj, inputs, params)
            [outputs{1}, outputs{2}, outputs{3}] = vl_nnsplit(inputs{1}) ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnsplit(derOutputs{1}, derOutputs{2}, derOutputs{3}) ;
            derParams = {} ;
        end
        
    end
end
