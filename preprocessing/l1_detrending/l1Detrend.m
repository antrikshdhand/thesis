function l1Detrend(f, t, ampls, alphas, names, plottingOptions, exportOptions)

% FUNCTION NAME:
%   l1Detrend
%
% DESCRIPTION:
%   Applies L1 detrending to each time segment of a given spectrogram using
%   a range of alpha values. 
%   The function supports various plotting options to visualise the results 
%   including comparison spectrograms, l1 trends of a random time segment, 
%   detrended versions of a random time segment, and 3D surfaces. 
%   The detrended spectrograms can also be exported as CSV files.
%
% INPUT:
%   f - (double array) Frequency values of the spectrogram
%   t - (double array) Time values of the spectrogram
%   ampls - (double matrix) Original amplitude spectrogram (f x t)
%   alphas - (double array) Array of alpha values for L1 detrending
%   names - (struct) Metadata for naming outputs, including:
%       * fileName - (string) Name of the input file
%       * className - (string) Class/category name for the input file
%   plottingOptions - (struct) Settings for plotting, including:
%       * plotSpec - (logical) Plot original and detrended spectrograms at 
%           various alpha values
%       * plotSegTrend - (logical) Plot a random time segment overlaid with
%           its L1 trend at various alpha values
%       * plotSegDetrended - (logical) Plot a random time segment overlaid 
%           with its detrended versions at various alpha values
%       * plot3DSurface - (logical) Plot 3D surfaces for the original and 
%           detrended spectrograms at all alpha values
%   exportOptions - (struct) Settings for exporting the detrended 
%       spectrograms, including:
%       * exportSpecCsv - (logical) Enable CSV export of the detrended 
%           spectrograms
%       * exportSpecCsvPath - (string) Directory for CSV export
%       * exportSpecMat - (logical) Enable MAT export of the detrended 
%           spectrograms
%       * exportSpecMatPath - (string) Directory for MAT export
%
% OUTPUT:
%   No direct outputs. However, the function generates visualisations and 
%   exports CSV files based on the specified options.
%
% REVISION HISTORY:
%   2024/11/09 - Initial commit to Github.

    validateInputs(alphas, exportOptions);

    % Initialise cell arrays to store detrended amplitudes and l1 trends 
    % for each alpha
    ampls_detrended_ls = cell(1, length(alphas));
    l1_trends_ls = cell(1, length(alphas));

    for i = 1:length(alphas)
        ampls_detrended_ls{i} = zeros(size(ampls));
        l1_trends_ls{i} = zeros(size(ampls));
    end
    
        % Run the l1 detrending algorithm on each time segment t_i of the 
        % spectrogram
        num_time_bins = length(t);
        for i = 1:num_time_bins
            % Get the ith column/ith time segment of the spectrogram
            t_i = ampls(:, i); 
        
            % Perform l1 filtering on the ith time segment
            lambda_max = l1tf_lambdamax(t_i);
            for j = 1:length(alphas)
                alpha = alphas(j);
                t_i_detrended = l1tf(t_i, alpha * lambda_max);
                
                % Store the trend and detrended amplitude in cell arrays
                l1_trends_ls{j}(:, i) = t_i_detrended;
                ampls_detrended_ls{j}(:, i) = t_i - t_i_detrended;
            end
        end
    
    % % Take the absolute value to prevent complex values
    % for i = 1:length(alphas)
    %     ampls_detrended_ls{i} = abs(ampls_detrended_ls{i});
    % end

    % Plot comparison spectrogram if specified
    if plottingOptions.plotSpec
        figure('units', 'normalized', 'outerposition', [0 0 1 1]);
        sgtitle( ...
            sprintf('%s (%s) - l1 detrend at various alpha', ...
            names.fileName, names.className), ...
            'Interpreter', 'None', ...
            'FontSize', 16, ...
            'FontWeight', 'bold');
        
        numPlotRows = ceil((length(alphas) + 1) / 2);
        subplot(numPlotRows, 2, 1);
        imagesc(f, t, ampls');
        xlabel('Frequency (Hz)');
        ylabel('Time');
        colormap('hot');
        colorbar;
        title('Original');
        for i = 1:length(alphas)
            subplot(numPlotRows, 2, i + 1);
            imagesc(f, t, ampls_detrended_ls{i}');
            xlabel('Frequency (Hz)');
            ylabel('Time');
            colormap('hot');
            colorbar;
            title(sprintf('alpha=%g', alphas(i)));
            clim([min(ampls(:)) max(ampls(:))]);
        end
    end

    random_time_seg = randi([1, num_time_bins]);

    % Plot one random time segment overlayed with its l1 trend at different
    % alpha values
    if plottingOptions.plotSegTrend
        numPlots = length(alphas) + 1;
        numCols = 2; 
        numRows = ceil(numPlots / numCols);

        % Plot original and l1 trends at different alphas
        figure('units', 'normalized', 'outerposition', [0 0 1 1]);
        sgtitle( ...
            sprintf("A random time segment (segment %i) overlaid with its " + ...
            "l1 trend at various alpha values\n%s (%s)", ...
            random_time_seg, names.fileName, names.className), ...
            'FontSize', 16, ...
            'Interpreter', 'None', ...
            'FontWeight', 'bold' ...
        );
        for i = 1:numPlots
            subplot(numRows, numCols, i);
            
            % Plot original spectrum in a semi-transparent blue
            plot(f, ampls(:, random_time_seg), 'LineWidth', 0.75, ...
                'Color', [0, 0, 1, 0.45]);
            hold on;
    
            % Plot l1 trend if not the first plot
            if i > 1
                plot(f, l1_trends_ls{i - 1}(:, random_time_seg), 'r-', ...
                    'LineWidth', 1.75);
                title(sprintf('Original and alpha=%g', alphas(i - 1)));
            else
                title('Original');
            end
    
            % Set plot labels and limits
            xlabel('Frequency (Hz)');
            ylabel('Amplitude');
            hold off;
        end
    end

    % Plot one random time segment overlaid with its detrended versions at 
    % different alpha values
    if plottingOptions.plotSegDetrended
        numPlots = length(alphas) + 1;
        numCols = 2;
        numRows = ceil(numPlots / numCols);

        figure('units', 'normalized', 'outerposition', [0 0 1 1]);
        sgtitle( ...
            sprintf("Random time segment %i overlaid with detrended versions " + ...
            "at various alpha values\n%s (%s)", ...
            random_time_seg, names.fileName, names.className), ...
            'FontSize', 16, ...
            'FontWeight', 'bold', ...
            'Interpreter', 'None' ...
        );
        
        for i = 1:numPlots
            subplot(numRows, numCols, i);
            plot(f, ampls(:, random_time_seg), 'LineWidth', 0.75, ...
                'Color', [0, 0, 1, 0.45]);
            hold on;
            if i > 1
                plot(f, ampls_detrended_ls{i - 1}(:, random_time_seg), ...
                    'r-', 'LineWidth', 1.75);
                title(sprintf('Original and alpha=%g', alphas(i - 1)));
            else
                title('Original');
            end
            xlabel('Frequency (Hz)');
            ylabel('Amplitude');
            hold off;
        end
    end

    % Plot 3D surfaces for original and detrended spectrograms at all alpha values
    if plottingOptions.plot3DSurface
        numPlots = length(alphas) + 1;
        numCols = 2;
        numRows = ceil(numPlots / numCols);

        figure('units', 'normalized', 'outerposition', [0 0 1 1]);
        sgtitle(sprintf('3D surfaces of original and detrended spectrograms\n%s (%s)', ...
            names.fileName, names.className), ...
            'FontSize', 16, ...
            'FontWeight', 'bold', ...
            'Interpreter', 'None' ...
        );

        % Plot original spectrogram
        subplot(numRows, numCols, 1);
        surf(f, t, ampls');
        xlabel('Frequency (Hz)');
        ylabel('Time');
        zlabel('Amplitude');
        title('Original 3D Surface');

        % Plot each detrended spectrogram for different alpha values
        for i = 1:length(alphas)
            subplot(numRows, numCols, i + 1);
            surf(f, t, ampls_detrended_ls{i}');
            xlabel('Frequency (Hz)');
            ylabel('Time');
            zlabel('Amplitude');
            title(sprintf('Detrended 3D surface (alpha=%g)', alphas(i)));
        end
    end

    % Export to CSV if specified
    if exportOptions.exportSpecCsv
        if ~isfolder(exportOptions.exportSpecCsvPath)
            mkdir(exportOptions.exportSpecCsvPath);
        end
        for i = 1:length(alphas)
            alpha_folder = fullfile(exportOptions.exportSpecCsvPath, "alpha_" + alphas(i));
            if ~isfolder(alpha_folder)
                mkdir(alpha_folder);
            end
            if ~isfolder(fullfile(alpha_folder, names.className))
                mkdir(fullfile(alpha_folder, names.className));
            end

            [~, name, ~] = fileparts(names.fileName); 
            csvFilename = fullfile( ...
                alpha_folder, ...
                names.className, ...
                [name, '.csv'] ...
            );
            writematrix(ampls_detrended_ls{i}, csvFilename);
        end
    end

    % Export to MAT if specified
    if exportOptions.exportSpecMat
        if ~isfolder(exportOptions.exportSpecMatPath)
            mkdir(exportOptions.exportSpecMatPath);
        end
        for i = 1:length(alphas)
            alpha_folder = fullfile(exportOptions.exportSpecMatPath, "alpha_" + alphas(i));
            if ~isfolder(alpha_folder)
                mkdir(alpha_folder);
            end
            if ~isfolder(fullfile(alpha_folder, names.className))
                mkdir(fullfile(alpha_folder, names.className));
            end

            [~, name, ~] = fileparts(names.fileName); 
            matFilepath = fullfile( ...
                alpha_folder, ...
                names.className, ...
                [name, '.mat'] ...
            );
            ampls_detrended_exp = ampls_detrended_ls{i};
            save(matFilepath, 'ampls_detrended_exp');
        end
    end

    function validateInputs(alphas, exportOptions)
        % Check alpha array length
        assert(length(alphas) < 8, ...
            'Too many alpha values; limit to less than 8.');

        % Check if export options are complete if exportSpecCsv is enabled
        if isfield(exportOptions, 'exportSpecCsv') && exportOptions.exportSpecCsv
            assert( ...
                isfield(exportOptions, 'exportSpecCsvPath') && ...
                ~isempty(exportOptions.exportSpecCsvPath), ...
                "exportSpecCsv is enabled, but no path provided in " + ...
                "exportSpecCsvPath." ...
            );
        end
    end
end