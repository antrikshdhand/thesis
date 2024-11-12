% This script exports the spectrogram PNGs of ONLY the vessels which have
% multiple recordings, as defined in `ships_with_multiple_recordings.csv` 
% which was extracted from the `organise_pairs.py` script.

% Same wavToSpec() options as baseline
FS = 5000;
windowLengthSec = 0.04;
windowLengthSeg = windowLengthSec * FS;
windowLengthSegRounded = 2^nextpow2(windowLengthSeg);

% Define processing options
processingOptions.resampleWav = false;
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
exportOptions.exportSpecPng = true;
exportOptions.exportWithAxes = false;
exportOptions.exportSpecPngPath = "specs_ships_multiple_recordings";
exportOptions.exportSpecCsv = false;
exportOptions.exportSpecMat = false;

% Point to the root directory of DeepShip dataset on your machine
rootDir = fullfile(getenv('USERPROFILE'), 'Desktop/thesis-ml/raw_datasets/deepship/DeepShip_5k_seg_3s');

VESSEL_CLASSES = {'Cargo', 'Passengership', 'Tanker', 'Tug'};

% Read in csv
csvFile = 'ships_with_multiple_recordings.csv';
multipleRecordingsCsv = readcell(csvFile, "NumHeaderLines", 1);
vesselsWithMultipleRecordings = multipleRecordingsCsv(:, 1);

close all;

for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.wav'));

    for j = 1:1000%length(classDir)
        currentFile = classDir(j).name;

        fileNameParts = split(currentFile, '-');
        vesselName = fileNameParts{1};
        
        if ~ismember(vesselName, vesselsWithMultipleRecordings)
            continue;
        end

        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        wavToSpec(vesselClass, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions);
    end
end