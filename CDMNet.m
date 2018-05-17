%Demo code for paper "COLOR IMAGE DEMOSAICKING USING A 3-STAGE CONVOLUTIONAL NEURAL NETWORK STRUCTURE"
%K. Cui, Z. Jin, E. Steinbach, Color Image Demosaicking using a 3-stage Convolutional Neural Network Structure,IEEE International Conference on Image Processing (ICIP 2018), Athens, Greece, Oktober 2018.
%Kai Cui <Kai.cui@tum.de>
%Lehrstuhl fuer Medientechnik
%Technische Universitaet Muenchen
%Last modified 17.05.2018
function net = CDMNet()
% matconvnet model
net          = dagnn.DagNN();
% Stage 1 layers, the dimention of the data is 1, G channel.
convRecon11  = dagnn.Conv('size',[3 3  1 128],'hasBias',true,'stride',[1,1],'pad',[1,1]);
convRecon12  = dagnn.Conv('size',[3 3 128 128],'hasBias',true,'stride',[1,1],'pad',[1,1]);
convRecon13  = dagnn.Conv('size',[3 3 128 1],'hasBias',true,'stride',[1,1],'pad',[1,1]);
% Stage 2 layers, the dimention of the data is 2, RG, GB channels.
convRecon21  = dagnn.Conv('size',[3 3  2 128],'hasBias',true,'stride',[1,1],'pad',[1,1]);
convRecon22  = dagnn.Conv('size',[3 3 128 128],'hasBias',true,'stride',[1,1],'pad',[1,1]);
convRecon23  = dagnn.Conv('size',[3 3 128  2],'hasBias',true,'stride',[1,1],'pad',[1,1]);
% Stage 3 layers, the dimention of the data is 3, RGB channels.
convRecon31  = dagnn.Conv('size',[3 3  3 128],'hasBias',true,'stride',[1,1],'pad',[1,1]);
convRecon32  = dagnn.Conv('size',[3 3 128 128],'hasBias',true,'stride',[1,1],'pad',[1,1]);
convRecon33  = dagnn.Conv('size',[3 3 128  3],'hasBias',true,'stride',[1,1],'pad',[1,1]);
%% -----Stage1 First Layer---------------
net.addLayer('S1SplitImage',dagnn.Split(),{'input'},{'input_R','input_G','input_B'});
net.addLayer('S1FirstLayer',convRecon11,{'input_G'},{'S1output_1'},{'S1para_1','S1bias_1'});
net.addLayer('S1relu1',dagnn.ReLU(),{'S1output_1'},{'S1output_1r'},{});
%% -----Stage1 Hidden Layers-------------
net.addLayer('S1HiddenLayer1',convRecon12,{'S1output_1r'},{'S1output_2'},{'S1para_2','S1bias_2'});
net.addLayer('S1BN1',dagnn.BatchNorm('numChannels', 128),{'S1output_2'},{'S1output_2bn'},{'S1para_2bn','S1bias_2bn','S1Momt_2bn'})
net.addLayer('S1relu2',dagnn.ReLU(),{'S1output_2bn'},{'S1output_2bnr'},{});

net.addLayer('S1HiddenLayer2',convRecon12,{'S1output_2bnr'},{'S1output_3'},{'S1para_3','S1bias_3'});
net.addLayer('S1BN2',dagnn.BatchNorm('numChannels', 128),{'S1output_3'},{'S1output_3bn'},{'S1para_3bn','S1bias_3bn','S1Momt_3bn'})
net.addLayer('S1relu3',dagnn.ReLU(),{'S1output_3bn'},{'S1output_3bnr'},{});

net.addLayer('S1HiddenLayer3',convRecon12,{'S1output_3bnr'},{'S1output_4'},{'S1para_4','S1bias_4'});
net.addLayer('S1BN3',dagnn.BatchNorm('numChannels', 128),{'S1output_4'},{'S1output_4bn'},{'S1para_4bn','S1bias_4bn','S1Momt_4bn'})
net.addLayer('S1relu4',dagnn.ReLU(),{'S1output_4bn'},{'S1output_4bnr'},{});
%% -----Stage1 Output--------------------
net.addLayer('S1output',convRecon13, {'S1output_4bnr'},{'S1prediction'},{'S1para_5','S1bias_5'});
net.addLayer('S1Sum1',dagnn.Sum(),{'S1prediction', 'input_G'},{'S1InterG'});

%% -----Stage2 First Layer---------------
net.addLayer('S2ConctLayer',dagnn.Concat(),{'input_R','S1InterG'}, {'S1InterRG'});
net.addLayer('S2ConctLayer1',dagnn.Concat(),{'input_R','input_G'}, {'input_RG'});
net.addLayer('S2FirstLayer',convRecon21,{'S1InterRG'},{'S2output_1'},{'S2para_1','S2bias_1'});
net.addLayer('S2relu1',dagnn.ReLU(),{'S2output_1'},{'S2output_1r'},{});
%% -----Stage2 Hidden Layers-------------
net.addLayer('S2HiddenLayer1',convRecon22,{'S2output_1r'},{'S2output_2'},{'S2para_2','S2bias_2'});
net.addLayer('S2BN1',dagnn.BatchNorm('numChannels', 128),{'S2output_2'},{'S2output_2bn'},{'S2para_2bn','S2bias_2bn','S2Momt_2bn'})
net.addLayer('S2relu2',dagnn.ReLU(),{'S2output_2bn'},{'S2output_2bnr'},{});

net.addLayer('S2HiddenLayer2',convRecon22,{'S2output_2bnr'},{'S2output_3'},{'S2para_3','S2bias_3'});
net.addLayer('S2BN2',dagnn.BatchNorm('numChannels', 128),{'S2output_3'},{'S2output_3bn'},{'S2para_3bn','S2bias_3bn','S2Momt_3bn'})
net.addLayer('S2relu3',dagnn.ReLU(),{'S2output_3bn'},{'S2output_3bnr'},{});

net.addLayer('S2HiddenLayer3',convRecon22,{'S2output_3bnr'},{'S2output_4'},{'S2para_4','S2bias_4'});
net.addLayer('S2BN3',dagnn.BatchNorm('numChannels', 128),{'S2output_4'},{'S2output_4bn'},{'S2para_4bn','S2bias_4bn','S2Momt_4bn'})
net.addLayer('S2relu4',dagnn.ReLU(),{'S2output_4bn'},{'S2output_4bnr'},{});
%% -----Stage2 Output--------------------
net.addLayer('S2output',convRecon23, {'S2output_4bnr'},{'S2prediction'},{'S2para_5','S2bias_5'});
net.addLayer('S2Sum1',dagnn.Sum(),{'S2prediction', 'S1InterRG'},{'S2InterRG'});

%% -----Stage3 First Layer---------------
net.addLayer('S3ConctLayer',dagnn.Concat(),{'S1InterG','input_B'}, {'S3InterGB'});
net.addLayer('S3ConctLayer1',dagnn.Concat(),{'input_G','input_B'}, {'input_GB'});
net.addLayer('S3FirstLayer',convRecon21,{'S3InterGB'},{'S3output_1'},{'S3para_1','S3bias_1'});
net.addLayer('S3relu1',dagnn.ReLU(),{'S3output_1'},{'S3output_1r'},{});
%% -----Stage3 Hidden Layers-------------
net.addLayer('S3HiddenLayer1',convRecon22,{'S3output_1r'},{'S3output_2'},{'S3para_2','S3bias_2'});
net.addLayer('S3BN1',dagnn.BatchNorm('numChannels', 128),{'S3output_2'},{'S3output_2bn'},{'S3para_2bn','S3bias_2bn','S3Momt_2bn'})
net.addLayer('S3relu2',dagnn.ReLU(),{'S3output_2bn'},{'S3output_2bnr'},{});

net.addLayer('S3HiddenLayer2',convRecon22,{'S3output_2bnr'},{'S3output_3'},{'S3para_3','S3bias_3'});
net.addLayer('S3BN2',dagnn.BatchNorm('numChannels', 128),{'S3output_3'},{'S3output_3bn'},{'S3para_3bn','S3bias_3bn','S3Momt_3bn'})
net.addLayer('S3relu3',dagnn.ReLU(),{'S3output_3bn'},{'S3output_3bnr'},{});

net.addLayer('S3HiddenLayer3',convRecon22,{'S3output_3bnr'},{'S3output_4'},{'S3para_4','S3bias_4'});
net.addLayer('S3BN3',dagnn.BatchNorm('numChannels', 128),{'S3output_4'},{'S3output_4bn'},{'S3para_4bn','S3bias_4bn','S3Momt_4bn'})
net.addLayer('S3relu4',dagnn.ReLU(),{'S3output_4bn'},{'S3output_4bnr'},{});
%% -----Stage3 Output--------------------
net.addLayer('S3output',convRecon23, {'S3output_4bnr'},{'S3prediction'},{'S3para_5','S3bias_5'});
net.addLayer('S3Sum1',dagnn.Sum(),{'S3prediction', 'S3InterGB'},{'S2InterGB'});
%% -----Split & reconcatenate--------------
net.addLayer('FinalSplit1',dagnn.Split_new(),{'S2InterRG'},{'final_R1','final_G1'});
net.addLayer('FinalSplit2',dagnn.Split_new(),{'S2InterGB'},{'final_G2','final_B2'});
%net.addLayer('FinalConct1',dagnn.Concat(),{'final_R1','S2InterGB'}, {'final_RGB1'});
net.addLayer('Finalconct2',dagnn.Concat(),{'S2InterRG','final_B2'}, {'final_RGB1'});
%net.addLayer('FinalOutput',dagnn.Sum(),{'final_RGB1', 'final_RGB2'},{'final_RGB'});

net.addLayer('S4FirstLayer',convRecon31,{'final_RGB1'},{'S4output_1'},{'S4para_1','S4bias_1'});
net.addLayer('S4relu1',dagnn.ReLU(),{'S4output_1'},{'S4output_1r'},{});
%% -----Stage3 Hidden Layers-------------
net.addLayer('S4HiddenLayer1',convRecon32,{'S4output_1r'},{'S4output_2'},{'S4para_2','S4bias_2'});
net.addLayer('S4BN1',dagnn.BatchNorm('numChannels', 128),{'S4output_2'},{'S4output_2bn'},{'S4para_2bn','S4bias_2bn','S4Momt_2bn'})
net.addLayer('S4relu2',dagnn.ReLU(),{'S4output_2bn'},{'S4output_2bnr'},{});

net.addLayer('S4HiddenLayer2',convRecon32,{'S4output_2bnr'},{'S4output_3'},{'S4para_3','S4bias_3'});
net.addLayer('S4BN2',dagnn.BatchNorm('numChannels', 128),{'S4output_3'},{'S4output_3bn'},{'S4para_3bn','S4bias_3bn','S4Momt_3bn'})
net.addLayer('S4relu3',dagnn.ReLU(),{'S4output_3bn'},{'S4output_3bnr'},{});

net.addLayer('S4HiddenLayer3',convRecon32,{'S4output_3bnr'},{'S4output_4'},{'S4para_4','S4bias_4'});
net.addLayer('S4BN3',dagnn.BatchNorm('numChannels', 128),{'S4output_4'},{'S4output_4bn'},{'S4para_4bn','S4bias_4bn','S4Momt_4bn'})
net.addLayer('S4relu4',dagnn.ReLU(),{'S4output_4bn'},{'S4output_4bnr'},{});
%% -----Stage3 Output--------------------
net.addLayer('S4output',convRecon33, {'S4output_4bnr'},{'S4prediction'},{'S4para_5','S4bias_5'});
net.addLayer('S4Sum1',dagnn.Sum(),{'S4prediction', 'final_RGB1'},{'final_RGB_final'});
%% ---------------Error------------------
net.addLayer('S1SplitLabel',dagnn.Split(),{'label'},{'label_R','label_G','label_B'});
net.addLayer('S2ConctLabel',dagnn.Concat(),{'label_R','label_G'}, {'label_RG'});
net.addLayer('S3ConctLabel',dagnn.Concat(),{'label_G','label_B'}, {'label_GB'});
net.addLayer('lossg',MSELoss(),{'S1InterG','label_G'},'LossG');
net.addLayer('lossrg',MSELoss(),{'S2InterRG','label_RG'},'LossRG');
net.addLayer('lossgb',MSELoss(),{'S2InterGB','label_GB'},'LossGB');
net.addLayer('lossrgb',MSELoss(),{'final_RGB_final','label'},'LossRGB');
net.addLayer('ErrorSum',dagnn.Sum(),{'LossG','LossRG','LossGB','LossRGB'},{'LossAll'});
net.initParams();

end