% This is an example of how you would use the l1Detrend() function.
% This script performs l1 detrending on a few samples of the DeepShip dataset, 
% located in `rootDir` (which contains 4 class folders).

addpath("../convert_wav_to_spec");
addpath("l1_algorithm")

%%% wavToSpec options %%%

% Define processing options
processingOptions.resampleWav = false;
processingOptions.normaliseWav = true;

% Define spectrogram settings
spectrogramOptions.window = hamming(256);
spectrogramOptions.noverlap = 128;
spectrogramOptions.nfft = 512;
spectrogramOptions.lowFreqCutoff = true;
spectrogramOptions.lowFreqCutoffIdx = 5; % Change this empirically
spectrogramOptions.highFreqCutoff = true;
spectrogramOptions.stopHz = 1000; % Change this based on your needs
spectrogramOptions.normaliseSpec = true;

% Define file paths and export options
exportOptions.plotSpec = false;
exportOptions.exportSpecPng = false;
exportOptions.exportSpecCsv = false;

%%% Perform l1 detrending %%%

rootDir = fullfile(getenv('USERPROFILE'), 'Desktop/acml_2024_s2/raw_datasets/deepship/DeepShip_5k_seg_3s');
VESSEL_CLASSES = {'Tug', 'Cargo', 'Passengership', 'Tanker'};

for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.wav'));
    
    for j = 1:1%length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        [ampls, P, f, t] = wavToSpec(vesselClass, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions);

        alphas = [1e-2, 1e-3, 1e-4];

        %%% l1Detrend options %%%
        l1PlottingOptions.plotSpec = true;
        l1PlottingOptions.plotSegTrend = true;
        l1PlottingOptions.plotSegDetrended = true;
        l1PlottingOptions.plot3DSurface = true;
        l1ExportOptions.exportSpecCsv = true;
        l1ExportOptions.exportSpecCsvPath = "detrended_specs";
        names.fileName = currentFile;
        names.className = vesselClass;

        l1Detrend(f, t, ampls, alphas, names, l1PlottingOptions, l1ExportOptions);
        %l1Detrend(f, t, P, alphas, true, true, names, l1ExportOptions);
    end
end