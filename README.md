<br />
<div align="center">
    <img src="docs/emblem.png" alt="Logo" width="150">
    <h3 align="center">Experiments in preprocessing techniques for underwater acoustic target recognition</h3>
    <p align="center">
        Official repository for my undergraduate thesis
        <br/>
        Supervised by Daniel La Mela (Thales) and Dr. Dong Yuan (USYD)
        <br/>
        <a href="docs/thesis.pdf">Dissertation</a>
        ·
        <a href="">Presentation</a>
    </p>
</div>

### Abstract

_Underwater acoustic target recognition (UATR) is a critical task in the application of sonar systems that aims to classify objects based on their acoustic signatures. Traditionally, UATR has relied on rule-based systems and the expertise of highly-trained sonar technicians to extract and classify features from raw sonar signals. However, recent advancements in artificial intelligence, particularly the rise of deep learning, have spurred interest in automating this process._

_A key factor influencing classification accuracy in machine learning models is the quality of the input data. To this end, various preprocessing techniques have been developed to enhance sonar signal quality by reducing noise and highlighting relevant features. This thesis evaluates the impact of three preprocessing techniques -- normalisation, detrending, and denoising -- on UATR performance, using the DeepShip dataset and a hybrid convolutional neural network-long short-term memory (CNN-LSTM) model as the experimental foundation._

_Experiments with normalisation revealed minimal impact, largely due to the inherent consistency of the dataset and prior preprocessing steps, such as power spectrogram conversion. Detrending with $\ell_1$ algorithms consistently reduced classification accuracy, likely due to over-smoothing and disruption of spectrogram features critical for the CNN-LSTM model. Efforts to adapt the Noise2Noise framework for denoising underwater spectrograms highlighted the challenges of dynamic underwater environments, where its assumptions could not be effectively met. Despite these limitations, masking-based denoising techniques showed promise in isolating regions of interest in spectrograms, offering a viable direction for future exploration._

_This thesis underscores the unique challenges of adapting machine learning techniques to the underwater acoustic domain. The findings highlight the need for tailored preprocessing and model development to address the inherent variability and complexity of sonar data, paving the way for more robust and effective UATR systems._

### How to use this repository

All code for the experiments run in the thesis can be found in this repository. You can either view the runner scripts (`ml/main/main_[experiment_name].ipynb`) if you want to recreate the experiments yourself, OR you can view the saved Jupyter Notebook files if you simply want to view the raw results presented in the thesis (`ml/main/models/saved/[experiment_name]_[date]`).

- Chapter 3 (baseline performance)
    - `preprocessing/wav_to_spec`: Program used to produce spectrograms given certain FFT parameters.
    - `preprocessing/utils`: Diagnostic plots presented in the chapter.
    - `ml/main/saved/cnn_lstm_baseline_20112024`: Baseline $k$-fold cross validation experiment run on DeepShip spectrograms.
- Chapter 4 (normalisation):
    - `preprocessing/normalisation`: MATLAB code for global and channel-based normalisation.
    - `ml/main/saved/cnn_lstm_channel_normalised` and `ml/main/saved/cnn_lstm_global_normalised`: Normalisation experiment results.
- Chapter 5 (detrending):
    - `preprocessing/l1_detrending`: MATLAB code for $\ell_1$ detrending implementation, built on the author's original codebase.
    - `ml/main/saved/cnn_lstm_detrended_24112024`: Detrending experiment results.
- Chapter 6 (denoising):
    - `ml/main/saved/n2n_imagenet10k_25112024`: Recreation of Noise2Noise paper.
    - `ml/main/saved/diff_spec_denoiser_05122024`: Approximation of N2N on spectrograms.
    - `ml/unet_segmentation_26112024`: Spectrogram masking experiment.

### Contact

This dissertation was undertaken in partial fulfilment of the requirements for the degree of Bachelor of Engineering Honours (Software). It was completed through the Engineering Sydney Industry Placement Scholarship (ESIPS) at Thales Under Water Systems from June - December 2024.

See below for contact details.


| Name           | Email                   |
|----------------|-------------------------|
| Antriksh Dhand  | dhandantriksh@gmail.com |
| Daniel La Mela (Thales) | Daniel.LaMela@thalesgroup.com.au |
| Dr. Dong Yuan (USYD) | dong.yuan@sydney.edu.au
