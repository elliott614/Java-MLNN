function [happyRows,unhappyRows] = P2Preprocessing(folder,dim,testInd,trainInd)
%P2PREPROCESSING Main function to perform preprocessing for P2 in CS540
%   folder = folder containing training images
%   dim = size of processessed images e.g. [36 26]
%   testInd = indeces of test data e.g. (1:20)
%   trainInd = indeces of train data e.g. (21:200)
%
%   example use:
%   P2Preprocessing('L:\My Documents\CS540\W2\P2\Training',[36 26],(1:20),(21:200));

%Read Training Data
happyFaces = uint8(zeros([dim 200]));
unhappyFaces = happyFaces;
for n = 1:100
    happyFaces(:,:,n) = imresize(imread(strcat(folder,'\',int2str(n),'b.jpg')), dim);
    unhappyFaces(:,:,n) = imresize(imread(strcat(folder,'\',int2str(n),'a.jpg')), dim);
end
for n = 101:200
    happyFaces(:,:,n) = rgb2gray(imresize(imread(strcat(folder,'\',int2str(n),'b.jpg')), dim));
    unhappyFaces(:,:,n) = rgb2gray(imresize(imread(strcat(folder,'\',int2str(n),'a.jpg')), dim));
end

%Reshape for writing to .csv files
happyRows = zeros(200,dim(1) * dim(2));
unhappyRows = happyRows;
for n = 1:200
    happyRows(n,:) = reshape(happyFaces(:,:,n).',1,[]);
    unhappyRows(n,:) = reshape(unhappyFaces(:,:,n).',1,[]);
end

%Write .csv files
csvwrite(strcat(folder,'\happyTraining.csv'),happyRows(trainInd,:));
csvwrite(strcat(folder,'\unhappyTraining.csv'),unhappyRows(trainInd,:));
csvwrite(strcat(folder,'\happyTest.csv'),happyRows(testInd,:));
csvwrite(strcat(folder,'\unhappyTest.csv'),unhappyRows(testInd,:));
end