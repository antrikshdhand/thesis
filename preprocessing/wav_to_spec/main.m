% This is an example of how you would use the wavToSpec() function.
% This script converts the DeepShip dataset, located in `rootDir` (which 
% contains 4 class folders), into power spectrograms, and exports both
% the csvs and the spectrogram plots into the current working directory.

FS = 5000; % This is the sampling frequency of the input files
windowLengthSec = 0.04; % Longer windows = better f res but worse t res
windowLengthSeg = windowLengthSec * FS;
windowLengthSegRounded = 2^nextpow2(windowLengthSeg);

% Define processing options
processingOptions.resampleWav = false;
processingOptions.resamplingRate = 4000;
processingOptions.normaliseWav = true;

% Define spectrogram settings
spectrogramOptions.window = hann(windowLengthSegRounded);
spectrogramOptions.noverlap = 0.75 * windowLengthSegRounded;
spectrogramOptions.nfft = 1024;
spectrogramOptions.lowFreqCutoff = true;
spectrogramOptions.startHz = 50; % Change this empirically
spectrogramOptions.highFreqCutoff = true;
spectrogramOptions.stopHz = 1000; % Change this based on your needs
spectrogramOptions.normaliseSpec = false;
spectrogramOptions.resize = true;

% Define file paths and export options
exportOptions.plotSpec = false;
exportOptions.exportSpecPng = false; % Slows down program 100x if set
exportOptions.exportWithAxes = false;
exportOptions.exportSpecPngPath = "spec_imgs";
exportOptions.exportSpecCsv = false;
exportOptions.exportSpecCsvPath = "spec_csvs";
exportOptions.exportSpecMat = true;
exportOptions.exportSpecMatPath = "spec_mats";

% Point to the root directory of DeepShip dataset on your machine
rootDir = fullfile(getenv('USERPROFILE'), 'Desktop/acml_2024_s2/raw_datasets/deepship/DeepShip_5k_seg_3s');

VESSEL_CLASSES = {'Cargo', 'Passengership', 'Tanker', 'Tug'};

close all;

for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.wav'));
    
    parfor j = 1:length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        [~, P, ~, ~] = wavToSpec(vesselClass, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions);

        % disp('Shape of P:');
        % disp(size(P)); % (freq bins, time bins)
    end
end