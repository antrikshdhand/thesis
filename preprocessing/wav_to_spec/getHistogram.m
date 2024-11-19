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
spectrogramOptions.amplitudeCutoff = false;
spectrogramOptions.lowFreqCutoff = true;
spectrogramOptions.startHz = 50; % Change this empirically
spectrogramOptions.highFreqCutoff = true;
spectrogramOptions.stopHz = 1000; % Change this based on Nyquist-Shannon theorem
spectrogramOptions.resize = true;

% Define file paths and export options
exportOptions.plotSpec = false;
exportOptions.exportSpecPng = false; % Slows down program 100x if set
exportOptions.exportWithAxes = true;
exportOptions.exportSpecPngPath = "spec_imgs";
exportOptions.exportSpecCsv = false;
exportOptions.exportSpecCsvPath = "spec_csvs";
exportOptions.exportSpecMat = false;
exportOptions.exportSpecMatPath = "spec_mats";

% Point to the root directory of DeepShip dataset on your machine
rootDir = fullfile("../../raw_datasets/DeepShip_5k_seg_3s/");

VESSEL_CLASSES = {'Cargo', 'Passengership', 'Tanker', 'Tug'};
NUM_SPECS = 500; % Number of spectrograms to process per class

specs = cell(length(VESSEL_CLASSES), NUM_SPECS);

% Custom colors for histograms
custom_colors = struct( ...
    "Cargo", "#1f77b4", ...
    "Passengership", "#ff7f0e", ...
    "Tanker", "#2ca02c", ...
    "Tug", "#d62728" ...
);

x_limits = [-45, 35]; 
y_limits = [0, 2100000]; 

close all;

% Save all spectrograms in memory
for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i}
    classDir = dir(fullfile(rootDir, vesselClass, '*.wav'));

    for j = 1:min(NUM_SPECS, length(classDir))
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        [~, P, ~, ~] = wavToSpec(vesselClass, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions);

        specs{i, j} = P; 
    end
end

disp("Done loading spectrograms");

% Plot average amplitude histogram for each class
fig = figure('units', 'normalized', 'outerposition', [0 0 0.4 0.4]);
for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    classSpecs = specs(i, :);
    
    % Concatenate all spectrograms for this class
    combinedAmpls = [];
    for j = 1:NUM_SPECS
        if isempty(classSpecs{j})
            continue;
        end
        combinedAmpls = [combinedAmpls; classSpecs{j}(:)];
    end
    
    % Compute histogram for the combined amplitudes
    subplot(2, 2, i);
    histogram(combinedAmpls, 50, ...
        'FaceColor', custom_colors.(vesselClass), 'EdgeColor', 'k');
    
    xlim(x_limits);
    ylim(y_limits);
    
    title(['Average Amplitude - ', vesselClass], ...
        'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Amplitude (dB)', 'FontSize', 12);
    ylabel('Frequency', 'FontSize', 12);
end

% sgtitle('Average Amplitude Histograms for Each Vessel Class', ...
    % 'FontSize', 18, 'FontWeight', 'bold');

exportgraphics(fig, "examples/averageHistogram.pdf")