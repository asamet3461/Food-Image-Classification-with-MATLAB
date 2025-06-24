clc;
clear;
close all;

% 1. Download the dataset
fprintf("Downloading Example Food Image data set (77 MB)...\n")
filename = matlab.internal.examples.downloadSupportFile('nnet', ...
    'data/ExampleFoodImageDataset.zip');
fprintf("Done.\n")
%% 

% 2. Unzip the dataset
filepath = fileparts(filename);
dataFolder = fullfile(filepath,'ExampleFoodImageDataset');
unzip(filename, dataFolder);
%% 

% 3. Load images
imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
%% 

% 4. Check number of images per class
disp(countEachLabel(imds));
numClasses = numel(categories(imds.Labels));
%% 

% 5. Split data into training and test sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');
%% 

% 6. Resize images to fit the network input size
inputSize = [224 224 3];
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augTest = augmentedImageDatastore(inputSize, imdsTest);
%% 

% 7. Load MobileNetV2 model
net = mobilenetv2;
lgraph = layerGraph(net);
%% 

% 8. Replace the final layers for transfer learning
newLayers = [
    fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','new_softmax')
    classificationLayer('Name','new_classoutput')];

% Remove old classification layers
lgraph = removeLayers(lgraph, {'Logits','Logits_softmax','ClassificationLayer_Logits'});

% Add new layers
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'global_average_pooling2d_1','new_fc');
%% 

% 9. Set training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augTest, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% 

% 10. Train the network
trainedNet = trainNetwork(augTrain, lgraph, options);
%% 

% 11. Test the network and calculate accuracy
predictedLabels = classify(trainedNet, augTest);
trueLabels = imdsTest.Labels;

accuracy = sum(predictedLabels == trueLabels)/numel(trueLabels);
fprintf('Test Set Accuracy: %.2f%%\n', accuracy * 100);
%% 

% 12. Plot the confusion matrix
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix for Food Image Classification');
%% 

% 13. Show random test images and their predicted labels
idx = randperm(length(imdsTest.Files), 9);  % Select 9 random test images

figure;
for i = 1:9
    subplot(3,3,i)
    
    % Read image and resize
    img = readimage(imdsTest, idx(i));
    imgResized = imresize(img, inputSize(1:2));
    
    % Predict label
    label = classify(trainedNet, imgResized);
    
    % Display image and label
    imshow(img)
    title(['Predicted: ' char(label)], 'FontSize', 12)
end

sgtitle('Sample Test Images with Predicted Labels');
