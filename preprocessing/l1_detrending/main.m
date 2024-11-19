% This is an example of how you would use the l1Detrend() function.
% This script performs l1 detrending on a few samples of the DeepShip dataset, 
% located in `rootDir` (which contains 4 class folders).

addpath("../wav_to_spec");
addpath("l1_algorithm")

%%% wavToSpec options %%%

FS = 5000; 
windowLengthSec = 0.04;
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
spectrogramOptions.amplitudeCutoff = true;
spectrogramOptions.lowFreqCutoff = true;
spectrogramOptions.startHz = 50;
spectrogramOptions.highFreqCutoff = true;
spectrogramOptions.stopHz = 1000;
spectrogramOptions.resize = false;

% Define file paths and export options
exportOptions.plotSpec = false;
exportOptions.exportSpecCsv = false;
exportOptions.exportSpecPng = false;
exportOptions.exportSpecMat = false;

%%% Perform l1 detrending %%%

rootDir = fullfile("../../raw_datasets/DeepShip_5k_seg_3s/");
VESSEL_CLASSES = {'Cargo', 'Passengership', 'Tanker', 'Tug'};

close all;

for i = 1:1%length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.wav'));
    
    for j = 975%1:length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        [ampls, P, f, t] = wavToSpec(vesselClass, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions);

        alphas = [1, 1e-1, 1e-2];

        %%% l1Detrend options %%%
        l1PlottingOptions = struct();
        l1PlottingOptions.plotSpec = true;
        l1PlottingOptions.plotSegTrend = true;
        l1PlottingOptions.plotSegDetrended = true;
        l1PlottingOptions.plot3DSurface = true;

        l1ExportOptions = struct();
        l1ExportOptions.resizeSpec = true;
        l1ExportOptions.exportSpecCsv = false;
        l1ExportOptions.exportSpecCsvPath = "detrended_specs_csv";
        l1ExportOptions.exportSpecMat = false;
        l1ExportOptions.exportSpecMatPath = "detrended_specs_mat";
        
        names = struct();
        names.fileName = currentFile;
        names.className = vesselClass;

        %l1Detrend(f, t, ampls, alphas, names, l1PlottingOptions, l1ExportOptions);
        l1Detrend(f, t, P, alphas, names, l1PlottingOptions, l1ExportOptions);
    end
end