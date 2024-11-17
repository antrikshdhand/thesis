function [ampls, P, f, t] = wavToSpec(vesselClass, currentFilePath, ...
    processingOptions, spectrogramOptions, exportOptions)

% FUNCTION NAME:
%   wavToSpec
%
% DESCRIPTION:
%   Processes a WAV file to generate a spectrogram, applying resampling 
%   and optional frequency cutoffs. Exports the spectrogram 
%   as a PNG image and/or CSV file based on provided options.
%
% INPUT:
%   vesselClass - (string) The class of the vessel (e.g., "Cargo", "Tanker").
%   currentFilePath - (string) Path to the WAV file for processing.
%   processingOptions - (struct) WAV file processing settings including:
%       * resampleWav - (logical) Flag to enable resampling
%       * resamplingRate - (double) Target sampling rate if resampling is enabled
%       * normaliseWav - (logical) Flag to enable waveform normalisation
%   spectrogramOptions - (struct) Settings for spectrogram generation, including:
%       * window - (integer) Window size for spectrogram calculation
%       * noverlap - (integer) Overlap length for spectrogram
%       * nfft - (integer) FFT length
%       * amplitudeCutoff - (logical) Flag to apply low-amplitude cutoff
%       * lowFreqCutoff - (logical) Flag to apply low-frequency cutoff
%       * startHz - (integer) Frequency threshold for low-frequency cutoff
%       * highFreqCutoff - (logical) Flag to apply high-frequency cutoff
%       * stopHz - (double) Frequency threshold for high-frequency cutoff
%       * resize - (logical) Flag to resize spectrogram before export
%   exportOptions - (struct) Settings for export, including:
%       * exportSpecPng - (logical) Enable PNG export of the spectrogram
%       * exportSpecPngPath - (string) Directory for PNG export
%       * exportWithAxes - (logical) Whether to keep spectrogram axes
%           in the exported PNG file
%       * exportSpecCsv - (logical) Enable CSV export of the spectrogram
%       * exportSpecCsvPath - (string) Directory for CSV export
%       * plotSpec - (logical) Flag to plot the spectrogram
%       * exportSpecMat - (logical) Enable MAT export of the spectrogram
%       * exportSpecMatPath - (string) Directory for MAT export
%
% OUTPUT:
%   ampls - (double matrix) Regular amplitude spectrogram (f x t)
%   P - (double matrix) Power spectrogram data (f x t)
%   f - (double array) Frequency values of the spectrogram
%   t - (double array) Time values of the spectrogram
%
% REVISION HISTORY:
%   2024/11/07 - Initial commit to GitHub.
%   2024/11/08 - Removed `verbose` and `vesselClasses` function parameters
%   2024/11/17 - Removed normalisation option

    validateInputs(exportOptions);

    [wav, fs] = audioread(currentFilePath);
    
    if processingOptions.resampleWav
        wav = resample(wav, processingOptions.resamplingRate, fs);
        fs = processingOptions.resamplingRate;
    end
    
    if processingOptions.normaliseWav
        wav = wav / max(abs(wav));
    end
    
    % Compute raw spectrogram and power spectrogram
    [S, f, t] = spectrogram(wav, spectrogramOptions.window, ...
        spectrogramOptions.noverlap, spectrogramOptions.nfft, fs);
    ampls = abs(S);

    P = 10 * log10(ampls.^2 + 1e-8); % dB scale
    
    if spectrogramOptions.amplitudeCutoff
        low_threshold = -30;
        ampls(ampls < low_threshold) = low_threshold;
        P(P < low_threshold) = low_threshold;
    end
    
    % disp(['Max amplitude before cutoffs: ', num2str(max(ampls(:)))]);
    % disp(['Min amplitude before cutoffs: ', num2str(min(ampls(:)))]);
    %disp(max(P(:))); 
    %disp(min(P(:)));

    % Apply frequency cutoffs based on settings
    if spectrogramOptions.lowFreqCutoff
        startHzIdx = find(f >= spectrogramOptions.startHz, 1);
        if ~isempty(startHzIdx)
            f = f(startHzIdx:end);
            ampls = ampls(startHzIdx:end, :);
            P = P(startHzIdx:end, :);
        end
    end
    
    if spectrogramOptions.highFreqCutoff
        stopHzIdx = find(f >= spectrogramOptions.stopHz, 1);
        if ~isempty(stopHzIdx)
            f = f(1:stopHzIdx);
            ampls = ampls(1:stopHzIdx, :);
            P = P(1:stopHzIdx, :);
        end
    end

    % disp(['Max amplitude after cutoffs: ', num2str(max(ampls(:)))]);
    % disp(['Min amplitude after cutoffs: ', num2str(min(ampls(:)))]);

    % Resize spectrogram if enabled
    if spectrogramOptions.resize
        P = imresize(P, [192 192]);
    end

    % Plot the power spectrogram if enabled
    [~, name, ~] = fileparts(currentFilePath);
    plotTitle = sprintf('%s (%s) PS', name, vesselClass);

    if exportOptions.plotSpec
        plotSpectrogram(f, t, P', plotTitle, true, true);
    end

    % Export the power spectrogram if enabled
    if exportOptions.exportSpecPng
        if exportOptions.exportWithAxes
            spec = plotSpectrogram(f, t, P', plotTitle, false, true);
        else
            spec = plotSpectrogram(f, t, P', plotTitle, false, false);
        end

        pngDir = fullfile(exportOptions.exportSpecPngPath, vesselClass);
        if ~isfolder(pngDir)
            mkdir(pngDir);
        end
        
        pngName = strcat(name, '.png'); % e.g. "ADVENTURE_1-70-20171207_seg001.png"
        fullfile(exportOptions.exportSpecPngPath, vesselClass, pngName);
        exportgraphics(spec, fullfile(exportOptions.exportSpecPngPath, vesselClass, pngName));
    end
    
    % Export CSV if enabled
    if exportOptions.exportSpecCsv
        [~, name, ~] = fileparts(currentFilePath); 

        csvDir = fullfile(exportOptions.exportSpecCsvPath, vesselClass);
        if ~isfolder(csvDir)
            mkdir(csvDir);
        end

        csvName = strcat(name, '.csv');
        writematrix(P', fullfile(exportOptions.exportSpecCsvPath, vesselClass, csvName));
    end

    % Export as MAT if enabled
    if exportOptions.exportSpecMat
        [~, name, ~] = fileparts(currentFilePath); 

        matDir = fullfile(exportOptions.exportSpecMatPath, vesselClass);
        if ~isfolder(matDir)
            mkdir(matDir);
        end

        matName = strcat(name, '.mat');
        Ptrans = P';
        save(fullfile(exportOptions.exportSpecMatPath, vesselClass, matName), 'Ptrans');
    end

    function validateInputs(exportOptions)    
        % Check for PNG export settings
        if exportOptions.exportSpecPng
            assert(~isempty(exportOptions.exportSpecPngPath), ...
                "exportSpecPng is enabled, but no path provided in exportSpecPngPath.");
        end
    
        % Check for CSV export settings
        if exportOptions.exportSpecCsv
            assert(~isempty(exportOptions.exportSpecCsvPath), ...
                "exportSpecCsv is enabled, but no path provided in exportSpecCsvPath.");
        end
    end

    function spectrogram = plotSpectrogram(f, t, ampls, plotTitle, visible, showAxes)
        spectrogram = figure('Visible', visible);
        ax = axes(spectrogram); 
        imagesc(ax, f, t, ampls);
        colormap(ax, 'hot');
    
        if ~showAxes
            set(ax, 'XTick', [], 'YTick', [], 'Box', 'off'); 
        else
            xlabel(ax, 'Frequency (Hz)');
            ylabel(ax, 'Time');
            colorbar(ax);
            title(ax, plotTitle, 'Interpreter', 'none');
        end
    end
end

