import os
import librosa
from pystoi import stoi

# Aligns the lengths of two audio signals by truncating the longer one.
def align_audio_length(audio1, audio2):
    min_length = min(len(audio1), len(audio2))
    audio1_aligned = audio1[:min_length]
    audio2_aligned = audio2[:min_length]
    return audio1_aligned, audio2_aligned

def calculate_stoi(damaged_audio_path, clear_audio_path, sr=24000):
    try:
        # Read/load damaged and clear audio files
        damaged, fs_damaged = librosa.load(damaged_audio_path, sr=sr)
        clear, fs_clear = librosa.load(clear_audio_path, sr=sr)

        # Resample audio if sampling rates do not match
        if fs_damaged != fs_clear:
            damaged = librosa.resample(y=damaged, orig_sr=fs_damaged, target_sr=fs_clear)
            clear = librosa.resample(y=clear, orig_sr=fs_clear, target_sr=fs_damaged)

        # Align the lengths of the audio signals
        damaged, clear = align_audio_length(damaged, clear)

        # Calculate STOI score
        stoi_score = stoi(clear, damaged, fs_clear, extended=False)

        return stoi_score
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Path to the damaged audio file
    damaged_audio = "./AudioS/clear_audio.wav"

    # Path to the clear audio file
    clear_audio = "./Audios/clear_audio.wav"

    results = []
    # Calculate STOI score for the given audio files
    score = calculate_stoi(damaged_audio, clear_audio)
    if score is not None:
        print(f"STOI score for {clear_audio} and {damaged_audio}: {score}")
        
        results.append((os.path.basename(clear_audio), os.path.basename(damaged_audio), score))
