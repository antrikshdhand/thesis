% This code plots the average frequency spectrum of NUM_SAMPLES_TO_AVG
% audio files for each class in the DeepShip dataset. 

workRoot = fullfile(getenv('USERPROFILE'), 'Desktop\acml_2024_s2\raw_datasets\deepship\DeepShip_5k_seg_3s'); % Change this according to setup
VESSEL_CLASSES = {'Tug', 'Cargo', 'Passengership', 'Tanker'};

NUM_SAMPLES_TO_AVG = 100;   % Number of samples to average per vessel class
DC_OFFSET = 15;             % Number of low-frequency components to ignore
SAVE_FIG = false;

for i = 1:length(VESSEL_CLASSES)
    vesselClass = VESSEL_CLASSES{i};
    wavFileDir = dir(fullfile(workRoot, vesselClass, '*.wav'));
    
    % Initialize an array to store FFT results for averaging
    fft_matrix = [];

    sampleCount = min(NUM_SAMPLES_TO_AVG, length(wavFileDir)); % Limit to available files if less than numSamples
    for j = 1:sampleCount % Consider using parfor if sampleCount is large
        % Get the current file name and path
        current_file = wavFileDir(j).name;
        fname = fullfile(workRoot, vesselClass, current_file);
        
        % Read in the wav file
        [audio_signal, fs] = audioread(fname);

        % Compute the length of the audio signal
        N = length(audio_signal);
        
        % Compute the FFT of the signal
        Y = fft(audio_signal);
        
        % Ignore the DC component and very low frequencies
        Y = Y(DC_OFFSET:end);
        
        % Compute the two-sided spectrum and then the single-sided spectrum
        P2 = abs(Y / N);             % Two-sided spectrum
        P1 = P2(1:floor(N/2)+1);     % Single-sided spectrum
        
        % Multiply by 2 (except for DC and Nyquist components) to maintain power
        P1(2:end-1) = 2 * P1(2:end-1);
        
        % Store the single-sided spectrum in the matrix for averaging
        fft_matrix(:, j) = P1;  % Each column is one sample's spectrum
    end
    
    % Compute the average spectrum across samples
    avg_spectrum = mean(fft_matrix, 2);

    % Define the frequency domain
    f = fs * (0:(N/2)) / N;

    % Plot the frequency spectrum
    fig = figure();
    hold on;
    
    % Plot individual spectra with low opacity
    for j = 1:sampleCount
        plot(f, fft_matrix(:, j), 'Color', [0.7 0.7 0.7 0.3]); % Light gray with transparency
    end
    
    % Plot the average spectrum with a solid line
    plot(f, avg_spectrum, 'b', 'LineWidth', 1.5); % Bold blue line for the average
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
    title(sprintf('Average Amplitude Spectrum (%s)', vesselClass));
    grid on;
    hold off;
    
    if SAVE_FIG
        if ~isfolder("avg_spectra")
            mkdir("avg_spectra");
            saveas(fig, sprintf('avg_spectra/%s.png', vesselClass));
        end
    end
end
