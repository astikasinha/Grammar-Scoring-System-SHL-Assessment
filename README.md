# Grammar-Scoring-System-SHL-Assessment
Spoken English Grammar Scoring
Objective
To develop a system that predicts a grammar proficiency score (1 to 5) from spoken English audio recordings. This is achieved by combining audio-based and text-based (transcription) features for training a machine learning model.

Approach Overview
Audio files are transcribed using the Whisper ASR model.
Transcriptions are analyzed using LanguageTool to extract grammar-related features.
Audio features (like pitch, energy, and MFCCs) are extracted using librosa.
Combined features are used to train an XGBoost regression model to predict grammar scores.
Model is evaluated using standard regression metrics and a custom accuracy.
Pipeline Architecture
1. Preprocessing
Transcription via Whisper (base model, fp16 disabled for Colab CPU compatibility).
Grammar error detection using language_tool_python.
Audio feature extraction via librosa (duration, MFCC mean/std, pitch, etc.).
2. Feature Engineering
Feature Description
This project focuses on predicting grammar proficiency scores from spoken audio responses. The features used in the model are derived from both the raw audio signal and the transcribed text using a combination of speech recognition, natural language processing, and audio signal processing techniques.

1. Audio Features (Extracted using librosa)
These features capture the underlying acoustic and temporal properties of the audio recording. They help understand pronunciation clarity, articulation, rhythm, and energy which are indirectly linked to fluency and delivery.

Basic Audio Properties:
duration: Total duration of the audio (in seconds), capturing the overall length of spoken response.
zero_crossing_rate_mean / zero_crossing_rate_std: Measures how frequently the signal changes sign, useful for detecting voice activity and sharp transitions.
energy_mean / energy_std: Root Mean Square (RMS) energy; indicates loudness and consistency in speech delivery.
spectral_centroid_mean / spectral_centroid_std: Indicates where the "center of mass" of the spectrum is located; related to perceived brightness of sound.
spectral_rolloff_mean / spectral_rolloff_std: The frequency below which a certain percentage of the total spectral energy lies.
spectral_bandwidth_mean / spectral_bandwidth_std: Width of the frequency band; reflects voice dynamics.
Pitch Features:
pitch_mean / pitch_std: Average and variability in pitch (fundamental frequency); can capture prosody and intonation patterns.
pitch_max / pitch_min: Range of pitch values; higher variation may indicate expressive or erratic speech.
MFCCs (Mel-Frequency Cepstral Coefficients):
MFCCs mimic human auditory perception and are widely used in speech processing.
For each of the first 13 MFCC coefficients:
mfcc_i_mean and mfcc_i_std (i = 1 to 13): Mean and standard deviation across the audio timeline.
Total Audio Features: 11 (basic) + 4 (pitch) + 26 (MFCCs) = 41 features

2. Grammar Features (From transcribed text using Whisper + LanguageTool)
After converting the audio into text using OpenAI's Whisper model, grammar analysis is performed using the LanguageTool API.

num_words: Total number of words in the transcribed response.
num_sentences: Number of sentence boundaries detected.
avg_sentence_length: Mean number of words per sentence; indicates sentence complexity.
num_grammar_errors: Total grammar/spelling mistakes identified.
error_rate: Ratio of grammar errors to total words; captures correctness and fluency.
These features help evaluate the correctness, coherence, and complexity of the speaker's grammar.

Total Grammar Features: 5

Final Feature Count
Audio-Based Features: 41
Text-Based Grammar Features: 5
Total Combined Features: 46
These features were standardized before training to ensure balanced learning across scales.

3. Heuristic Grammar Scoring (for testing set)
A rule-based function maps grammar error stats to a discrete score (1.0 to 5.0).
4. Modeling Approach
The modeling pipeline is designed to learn the relationship between engineered audio/text features and the target grammar proficiency score (ranging from 1.0 to 5.0). Below are the key components of the modeling architecture:

Feature Normalization
Standardized all feature values using Scikit-learn's StandardScaler.
This ensures each feature contributes equally to the model training, especially important for algorithms like XGBoost.
Model Used: XGBoost Regressor
XGBoost is a powerful, tree-based ensemble learning algorithm known for its high performance and robustness to overfitting. It was chosen for its ability to handle nonlinear feature interactions and small-to-medium sized datasets effectively.

Model Parameters:
n_estimators: 400 — number of boosting rounds (trees).
max_depth: 7 — maximum depth of a tree to control complexity.
learning_rate: 0.03 — step size shrinkage used in update to prevent overfitting.
subsample: 0.8 — fraction of training instances used in each boosting round.
colsample_bytree: 0.8 — fraction of features used for constructing each tree.
random_state: 42 — ensures reproducibility.
Model Training and Validation
Data Split: 80% training / 20% validation
The model was trained using the preprocessed features and evaluated using regression metrics (RMSE, R², and custom accuracy).
Evaluation Results (Validation Set)
RMSE: ≈ 0.8413
R² Score: ≈ 0.4808
Custom Accuracy (within ±0.5 tolerance): 0.4607
Visualizations
Feature importance (top 20 features by gain).
Actual vs Predicted scatter plot.
Residual histogram (errors centered around 0).
Predicted vs Actual curves for first 100 samples.
Submission
Test audios are processed with the heuristic scoring function.
Predictions saved in submission.csv in the required format (filename, label).
