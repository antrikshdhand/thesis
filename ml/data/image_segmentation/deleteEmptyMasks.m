% Some masks exported by MATLAB Image Labeller are empty. This script
% will delete them.

maskDir = fullfile('data', 'PixelLabelData', 'pixelLabelData');
originalImagesDir = fullfile('data', 'specs_ships_multiple_recordings');

% List of all mask files
maskFiles = dir(fullfile(maskDir, 'Label_*.png'));

numTotalFiles = size(maskFiles, 1);
numDeletedFiles = 0;
for i = 1:length(maskFiles)
    maskFile = fullfile(maskDir, maskFiles(i).name);
    maskImage = imread(maskFile);

    % Delete the file if it's empty
    if all(maskImage(:) == 0)
        delete(maskFile);
        disp(['Deleted empty mask: ', maskFiles(i).name]);

        numDeletedFiles = numDeletedFiles + 1;
    end
end

disp('---- SUMMARY ----');
disp(['Number of initial files: ', num2str(numTotalFiles)]);
disp(['Number of files deleted: ', num2str(numDeletedFiles)]);
disp(['Number of files remaining: ', num2str(numTotalFiles - numDeletedFiles)]);