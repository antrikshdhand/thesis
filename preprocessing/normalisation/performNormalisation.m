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

stats = load('deepship_stats.mat');

disp('Applying and exporting normalised spectrograms...');

% APPLY NORMALISATION
for i = 1:1%length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.mat'));

    for j = 1:1%length(classDir)
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        % Load the spectrogram
        data = load(currentFilePath);
        spectrogram = data.Ptrans;

        % Apply global normalisation
        globalNormalised = (spectrogram - stats.globalMean) / stats.globalStd;

        % Save global-normalised spectrogram
        globalOutputPath = fullfile(outputDirGlobal, vesselClass, currentFile);
        if ~isfolder(fullfile(outputDirGlobal, vesselClass))
            mkdir(fullfile(outputDirGlobal, vesselClass));
        end
        save(globalOutputPath, 'globalNormalised');

        % Apply channel-based normalisation
        channelNormalised = (spectrogram - stats.fMeans) ./ stats.fStds;

        % Save channel-normalised spectrogram
        channelOutputPath = fullfile(outputDirChannel, vesselClass, currentFile);
        if ~isfolder(fullfile(outputDirChannel, vesselClass))
            mkdir(fullfile(outputDirChannel, vesselClass));
        end
        channelNormalised = real(channelNormalised);
        save(channelOutputPath, 'channelNormalised');
    end
end