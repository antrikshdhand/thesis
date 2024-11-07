% Define processing options
processingOptions.resampleWav = false;
processingOptions.resamplingRate = 4000;
processingOptions.normaliseWav = true;

% Define spectrogram settings
spectrogramOptions.window = hamming(256);
spectrogramOptions.noverlap = 128;
spectrogramOptions.nfft = 512;
spectrogramOptions.lowFreqCutoff = true;
spectrogramOptions.lowFreqCutoffIdx = 5; % Change this empirically
spectrogramOptions.highFreqCutoff = true;
spectrogramOptions.stopHz = 1000; % Change this based on your needs

% Define file paths and export options
exportOptions.plotSpec = false;
exportOptions.exportSpecPng = true;
exportOptions.exportSpecPngPath = "spec_imgs";
exportOptions.exportSpecCsv = true;
exportOptions.exportSpecCsvPath = "spec_csvs";

verbose = false;

% Point to the root directory of DeepShip dataset on your machine
rootDir = fullfile(getenv('USERPROFILE'), 'Desktop/acml_2024_s2/raw_datasets/deepship/DeepShip_5k_seg_3s');

VESSEL_CLASSES = {'Tug', 'Cargo', 'Passengership', 'Tanker'};

for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.wav'));
    
    for j = 1:1%length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        wavToSpec(vesselClass, VESSEL_CLASSES, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions, verbose);
    end
end