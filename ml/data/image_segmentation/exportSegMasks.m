EXPORT_FOLDER = 'masks';

specsPath = fullfile('data', 'spectrograms');
masksPath = fullfile('data', 'PixelLabelData', 'pixelLabelData');
exportPath = fullfile('data', EXPORT_FOLDER);

if ~isfolder(exportPath)
    mkdir(exportPath);
end

specsDir = dir(fullfile(specsPath, '*.png')); 
masksDir = dir(fullfile(masksPath, 'Label_*.png')); 

% Define categories for each pixel value
labelIDs = [1];
labelCategories = ["narrowband"];

close all;

for i = 1:length(masksDir)
    maskFile = fullfile(masksPath, masksDir(i).name);
    [~, maskFileName, ~] = fileparts(maskFile);
    parts = split(maskFileName, '_');
    index = parts{2};

    specFile = fullfile(specsPath, specsDir(str2double(index)).name);
    [~, specFileName, ~] = fileparts(specFile);

    disp(['Spec: ', specFileName]);
    disp(['Mask: ', maskFileName]);

    specImage = imread(specFile);
    maskImage = imread(maskFile);

    specCategories = categorical(maskImage, labelIDs, labelCategories);

    exportMask = uint8(specCategories == "narrowband") * 255; % 255 for label, 0 otherwise

    % figure;
    % imshow(labeloverlay(specImage, exportMask));

    % Save the exportMask
    exportFileName = fullfile(exportPath, [specFileName, '_mask.png']);
    imwrite(exportMask, exportFileName);
end
