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

rootDir = fullfile("../../raw_datasets/DeepShip_5k_seg_3s/");

VESSEL_CLASSES = {'Cargo', 'Passengership', 'Tanker', 'Tug'};

close all;

for i = 4
    vesselClass = VESSEL_CLASSES{i};
    classDir = dir(fullfile(rootDir, vesselClass, '*.wav'));

    for j = 6547
        currentFile = classDir(j).name;
        currentFilePath = fullfile(rootDir, vesselClass, currentFile);

        spectrogramOptions.amplitudeCutoff = false;

        [~, P1, f, t] = wavToSpec(vesselClass, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions);

        spectrogramOptions.amplitudeCutoff = true;

        [~, P2, f, t] = wavToSpec(vesselClass, currentFilePath, ...
            processingOptions, spectrogramOptions, exportOptions);

        % Original spectrogram
        fig = figure('units', 'normalized', 'outerposition', [0 0 0.4 0.33]);
        subplot(1, 2, 1);
        imagesc(f, t, P1');
        title('Before Cutoff', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Frequency (Hz)');
        ylabel('Time (s)');
        colorbar;
        colormap('hot');
        
        % Spectrogram after applying cutoff
        subplot(1, 2, 2);
        imagesc(f, t, P2');
        title('After Cutoff', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Frequency (Hz)', 'FontSize', 11);
        ylabel('Time (s)', 'FontSize', 11);
        colorbar;
        colormap('hot');

        exportgraphics(fig, "examples/amplCutoffComparison.pdf")
    end
end