rootDir = fullfile("../../ml/data/deepship_baseline_unnorm_mat/"); % Ptrans
VESSEL_CLASSES = {'Cargo', 'Passengership', 'Tanker', 'Tug'};

outputDirGlobal = 'global_normalised';
outputDirChannel = 'channel_normalised';
if ~isfolder(outputDirGlobal)
    mkdir(outputDirGlobal);
end
if ~isfolder(outputDirChannel)
    mkdir(outputDirChannel);
end

FREQ_BINS = 192;
TIME_BINS = 192;
SPECTROGRAM_DIMENSIONS = FREQ_BINS * TIME_BINS;

spectrogramSums = 0; % Running sum of all spectrogram values
squaredDiffSum = 0;
numSpectrograms = 0;

% GET GLOBAL MEAN %
disp('Getting global mean...');
for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.mat'));

    for j = 1:5%length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        data = load(currentFilePath);
        spectrogram = data.Ptrans;

        numSpectrograms = numSpectrograms + 1;
        spectrogramSums = spectrogramSums + sum(spectrogram, "all");
    end
end

globalMean = spectrogramSums / (SPECTROGRAM_DIMENSIONS * numSpectrograms);

disp('Getting global stddev...');

% GET GLOBAL STANDARD DEVIATION %
for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.mat'));

    for j = 1:length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        data = load(currentFilePath);
        spectrogram = data.Ptrans;

        squaredDiffSum = squaredDiffSum + sum((spectrogram - globalMean).^2, "all");
    end
end

globalStd = sqrt(squaredDiffSum / (SPECTROGRAM_DIMENSIONS * numSpectrograms));

disp('Getting channel means and stddevs...');

% GET CHANNEL MEANS AND STDDEVS FOR ALL FREQUENCY BINS %
fSums = zeros(FREQ_BINS, 1);
fSquaredSum = zeros(FREQ_BINS, 1); 

for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.mat'));

    for j = 1:length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        data = load(currentFilePath);
        spectrogram = data.Ptrans;

        fSums = fSums + sum(spectrogram, 2); 
        fSquaredSum = fSquaredSum + sum(spectrogram.^2, 2); 
    end
end

fMeans = fSums / (TIME_BINS * numSpectrograms);
fVariances = (fSquaredSum / (TIME_BINS * numSpectrograms)) - (fMeans.^2);
fStds = sqrt(fVariances);

disp('Applying and exporting normalised spectrograms...');

% APPLY NORMALISATION
for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.mat'));

    for j = 1:length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        % Load the spectrogram
        data = load(currentFilePath);
        spectrogram = data.Ptrans;

        % Apply global normalisation
        globalNormalised = (spectrogram - globalMean) / globalStd;

        % Save global-normalised spectrogram
        globalOutputPath = fullfile(outputDirGlobal, vesselClass, currentFile);
        if ~isfolder(fullfile(outputDirGlobal, vesselClass))
            mkdir(fullfile(outputDirGlobal, vesselClass));
        end
        save(globalOutputPath, 'globalNormalised');

        % Apply channel-based normalisation
        channelNormalised = (spectrogram - fMeans) ./ fStds;

        % Save channel-normalised spectrogram
        channelOutputPath = fullfile(outputDirChannel, vesselClass, currentFile);
        if ~isfolder(fullfile(outputDirChannel, vesselClass))
            mkdir(fullfile(outputDirChannel, vesselClass));
        end
        save(channelOutputPath, 'channelNormalised');
    end
end
