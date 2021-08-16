clc
clear all;
close all;


%folderloc = 'G:\signature';
folderloc = "G:\Signature_Set3\BHSig260";
imds = imageDatastore(folderloc, ...
    'IncludeSubfolders', true,...
    'LabelSource', 'none')

files = imds.Files
imds
parts = split(files,filesep)
%basefile = fullfile(parts(1, 1), parts(1,2), filesep)

for i = 1:length(files)
    filess = string(files);
    image = imread(filess(i));
    [x,y,z] = size(image)
    %image = im2gray(image);
    resize = imresize(image,[224 224]);
    imwrite(resize,filess(i))
    
    %fullfile('G:\signature', parts(i,3:4))
    
end

