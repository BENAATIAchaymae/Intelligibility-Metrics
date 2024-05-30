# Metrics
Speech Quality Metrics

This repository contains Python scripts to calculate Speech Quality Metrics including Short-Time Objective Intelligibility (STOI) and Word Error Rate (WER) between two sets of audio and transcription files.

Requirements:

- Python 3.x
- Required Python packages:
  - librosa
  - pystoi
  - jiwer

Install the required packages using pip:
  pip install librosa pystoi jiwer

STOI Calculation:

- align_audio_length Function:

This function aligns the lengths of two audio signals by truncating the longer one to match the length of the shorter one.

- calculate_stoi Function:

This function calculates the STOI score between a damaged audio file and a clear audio file:

- It reads the audio files using librosa.
- If the sampling rates of the audio files do not match, it resamples them.
- It aligns the lengths of the audio signals using align_audio_length.
- It computes the STOI score using the stoi function from pystoi.

Running the STOI Script:

Ensure that your damaged and clear audio files are in the correct directory and update the paths in the script accordingly.

  # Path to the damaged audio file
  damaged_audio = "./Audio/damaged_audio.wav"

  # Path to the clear audio file
  clear_audio = "./Audio/clear_audio.wav"

To run the script, execute the following command:

  python calculate_stoi.py

The script will print the STOI score for the specified audio files.

WER Calculation:

- calculate_wer Function:

This function calculates the Word Error Rate (WER) between a reference transcript and a hypothesis transcript using the wer function from jiwer.

Running the WER Script:

Ensure that your reference and hypothesis transcripts are in the correct format and directory. Update the paths in the script accordingly.

  # Path to the reference transcript file
  ref_transcript = "./Transcripts/ref_transcript.txt"

  # Path to the hypothesis transcript file
  hyp_transcript = "./Transcripts/hyp_transcript.txt"

To run the script, execute the following command:

  python calculate_wer.py

The script will print the Word Error Rate (WER) between the reference and hypothesis transcripts.

Authors:
BENAATIA Chaymae
