function [ampls, P, f, t] = wavToSpec(vesselClass, currentFilePath, ...
    processingOptions, spectrogramOptions, exportOptions)

% FUNCTION NAME:
%   wavToSpec
%
% DESCRIPTION:
%   Processes a WAV file to generate a spectrogram, applying resampling, 
%   normalisation, and optional frequency cutoffs. Exports the spectrogram 
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
%       * lowFreqCutoff - (logical) Flag to apply low-frequency cutoff
%       * lowFreqCutoffIdx - (integer) Start index for low-frequency cutoff
%       * highFreqCutoff - (logical) Flag to apply high-frequency cutoff
%       * stopHz - (double) Frequency threshold for high-frequency cutoff
%       * normaliseSpec - (logical) Flag to apply 0-1 normalisation to the
%           spectrogram
%   exportOptions - (struct) Settings for export, including:
%       * exportSpecPng - (logical) Enable PNG export of the spectrogram
%       * exportSpecPngPath - (string) Directory for PNG export
%       * exportSpecCsv - (logical) Enable CSV export of the spectrogram
%       * exportSpecCsvPath - (string) Directory for CSV export
%       * plotSpec - (logical) Flag to plot the spectrogram
%
% OUTPUT:
%   ampls - (double matrix) Regular amplitude spectrogram
%   P - (double matrix) Power spectrogram data
%   f - (double array) Frequency values of the spectrogram
%   t - (double array) Time values of the spectrogram
%
% REVISION HISTORY:
%   2024/11/07 - Initial commit to GitHub.
%   2024/11/08 - Removed `verbose` and `vesselClasses` function parameters

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
    P = 10 * log10(ampls.^2 + 1e-8); 

    % Apply frequency cutoffs based on settings
    if spectrogramOptions.lowFreqCutoff
        f = f(spectrogramOptions.lowFreqCutoffIdx:end);
        ampls = ampls(spectrogramOptions.lowFreqCutoffIdx:end, :);
        P = P(spectrogramOptions.lowFreqCutoffIdx:end, :);
    end
    
    if spectrogramOptions.highFreqCutoff
        stopHzIdx = find(f >= spectrogramOptions.stopHz, 1);
        if ~isempty(stopHzIdx)
            f = f(1:stopHzIdx);
            ampls = ampls(1:stopHzIdx, :);
            P = P(1:stopHzIdx, :);
        end
    end

    % 0-1 normalisation of the spectrograms
    if spectrogramOptions.normaliseSpec
        ampls = (ampls - min(ampls(:))) / (max(ampls(:)) - min(ampls(:)));
        P = (P - min(P(:))) / (max(P(:)) - min(P(:)));
    end
    
    % Plot the spectrogram if enabled
    if exportOptions.plotSpec || exportOptions.exportSpecPng
        [~, name, ~] = fileparts(currentFilePath);
        plotTitle = sprintf('%s (%s) PS', name, vesselClass);
        spec = plotSpectrogram(f, t, P', plotTitle, exportOptions.plotSpec);

        if exportOptions.exportSpecPng
            pngName = fullfile(vesselClass, [name, '.png']); % e.g. "Cargo/ADVENTURE_1-70-20171207_seg001.png"
    
            if ~isfolder(exportOptions.exportSpecPngPath)
                mkdir(exportOptions.exportSpecPngPath);
                mkdir(fullfile(exportOptions.exportSpecPngPath, vesselClass));
            end
    
            exportgraphics(spec, fullfile(exportOptions.exportSpecPngPath, pngName));
        end
    end
    
    % Export CSV if enabled
    if exportOptions.exportSpecCsv
        [~, name, ~] = fileparts(currentFilePath); 
        csvName = fullfile(vesselClass, [name, '.csv']);
        
        if ~isfolder(exportOptions.exportSpecCsvPath)
            mkdir(exportOptions.exportSpecCsvPath);
            mkdir(fullfile(exportOptions.exportSpecCsvPath, vesselClass));
        end
    
        writematrix(P', fullfile(exportOptions.exportSpecCsvPath, csvName));
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
end

function spectrogram = plotSpectrogram(f, t, ampls, plotTitle, visible)
    spectrogram = figure('Visible', visible);
    imagesc(f, t, ampls);
    xlabel('Frequency (Hz)');
    ylabel('Time');
    colormap('hot');
    colorbar;
    title(plotTitle, 'Interpreter', 'none');
end
