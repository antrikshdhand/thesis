% Script to compare original, global normalised, and channel normalised spectrograms

% Directories
rootDir = fullfile("../../ml/data/deepship_baseline_unnorm_mat"); % Ptrans
% globalDir = fullfile('../../ml/data/deepship_baseline_global_normalised');
globalDir = fullfile('global_normalised/');
channelDir = fullfile('../../ml/data/deepship_baseline_channel_normalised');

VESSEL_CLASSES = {'Cargo', 'Passengership', 'Tanker', 'Tug'};

close all;

% Loop through each vessel class
for i = 1:1%length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.mat'));

    for j = 1:1%length(classDir)
        % Read the current file
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        % Load original, global-normalised, and channel-normalised spectrograms
        data = load(currentFilePath);
        originalSpectrogram = data.Ptrans;

        globalPath = fullfile(globalDir, vesselClass, currentFile);
        globalData = load(globalPath);
        globalNormalisedSpectrogram = globalData.globalNormalised;

        globalNormalisedSpectrogram(:, 1)

        channelPath = fullfile(channelDir, vesselClass, currentFile);
        channelData = load(channelPath);
        channelNormalisedSpectrogram = channelData.channelNormalised;

        % Compute histograms for comparison
        fig1 = figure('units', 'normalized', 'outerposition', [0 0 0.5 0.45]);

        subplot(1, 3, 1);
        histogram(originalSpectrogram(:), 50, 'FaceColor', [0.1 0.5 0.8]);
        title('Original Spectrogram', 'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Amplitude', 'FontSize', 13);
        ylabel('Frequency', 'FontSize', 13);
        grid on;

        subplot(1, 3, 2);
        histogram(globalNormalisedSpectrogram(:), 50, 'FaceColor', [0.8 0.4 0.1]);
        title('Global Normalised', 'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Amplitude', 'FontSize', 13);
        ylabel('Frequency', 'FontSize', 13);
        grid on;

        subplot(1, 3, 3);
        histogram(channelNormalisedSpectrogram(:), 50, 'FaceColor', [0.1 0.8 0.4]);
        title('Channel Normalised', 'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Amplitude', 'FontSize', 13);
        ylabel('Frequency', 'FontSize', 13);
        grid on;

        % % sgtitle( ...
        % %     sprintf('Amplitude Distributions - %s (%s)', currentFile, vesselClass), ...
        % %     'FontSize', 14, ...
        % %     'FontWeight', 'bold', ...
        % %     'Interpreter', 'none' ...
        % % );
        % 
        exportgraphics(fig1, "examples/histogramComparison.pdf")

        % Plot the original, global normalised, and channel normalised spectrograms
        fig2 = figure('units', 'normalized', 'outerposition', [0 0 0.5 0.45]);

        subplot(1, 3, 1);
        imagesc(originalSpectrogram);
        colorbar;
        colormap('hot');
        title('Original Spectrogram', 'FontSize', 16);
        xlabel('Frequency Bins', 'FontSize', 13);
        ylabel('Time Bins', 'FontSize', 13);

        subplot(1, 3, 2);
        imagesc(globalNormalisedSpectrogram);
        colorbar;
        colormap('hot');
        title('Global Normalised', 'FontSize', 16);
        xlabel('Frequency Bins', 'FontSize', 13);
        ylabel('Time Bins', 'FontSize', 13);

        subplot(1, 3, 3);
        imagesc(channelNormalisedSpectrogram);
        colorbar;
        colormap('hot');
        title('Channel Normalised', 'FontSize', 16);
        xlabel('Frequency Bins', 'FontSize', 13);
        ylabel('Time Bins', 'FontSize', 13);

        % sgtitle( ...
        %     sprintf('Normalisation Techniques - %s (%s)', currentFile, vesselClass), ...
        %     'FontSize', 14, ...
        %     'FontWeight', 'bold', ...
        %     'Interpreter', 'none' ...
        % );

        exportgraphics(fig2, "examples/spectrogramComparison.pdf")

    end
end
