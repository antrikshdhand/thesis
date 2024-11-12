specsPath = fullfile('data', 'spectrograms');
masksPath = fullfile('data', 'PixelLabelData', 'pixelLabelData');

specsDir = dir(fullfile(specsPath, '*.png')); 
masksDir = dir(fullfile(masksPath, 'Label_*.png')); 

% Define categories for each pixel value
% (https://www.mathworks.com/help/releases/R2024a/vision/ug/labeler-pixel-label-data-storage.html)
labelIDs = [1];
labelCategories = ["narrowband"];

close all;

for i = 10:15%length(masksDir)
    
    maskFile = fullfile(masksPath, masksDir(i).name);
    [~, maskFileName, ~] = fileparts(maskFile);
    parts = split(maskFileName, '_');
    index = str2double(parts{2});
    
    % Get corresponding spec file
    specFile = fullfile(specsPath, specsDir(index).name);
    [~, specFileName, ~] = fileparts(specFile);

    disp(['Spec: ', specFileName])
    disp(['Mask: ', maskFileName])

    specImage = imread(specFile);
    maskImage = imread(maskFile);

    specCategories = categorical(maskImage, labelIDs, labelCategories);

    figure
    imshow(labeloverlay(specImage, maskImage))
end
