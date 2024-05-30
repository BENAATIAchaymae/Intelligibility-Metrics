import os
import torchaudio
import torch
import jiwer
import pandas as pd

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

if __name__ == "__main__":
    # Specify the paths to the clean and noisy audio files
    clean_audio = './Audios/clear_audio.wav'
    noisy_audio = './Audios/damaged_audio.wav'

    error_scores = []


    # Initialize the Wav2Vec2 model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = bundle.get_model().to(device)
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())

    results = []

    # Load the clean and noisy audio files
    clean_waveform, _ = torchaudio.load(clean_audio)
    noisy_waveform, _ = torchaudio.load(noisy_audio)

    with torch.inference_mode():
        # Get the emission probabilities from the model for both clean and noisy audio
        clean_emission, _ = model(clean_waveform.to(device))
        clean_transcript = decoder(clean_emission[0])

        noisy_emission, _ = model(noisy_waveform.to(device))
        noisy_transcript = decoder(noisy_emission[0])

    # Preprocess transcripts to remove punctuation and special characters
    reference = jiwer.RemovePunctuation()(clean_transcript)
    hypothesis = jiwer.RemovePunctuation()(noisy_transcript)
    
    reference = reference.replace('|', ' ')
    hypothesis = hypothesis.replace('|', ' ')

    # Define custom transformations for WER calculation
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )

    # Calculate Word Error Rate (WER)
    error = jiwer.wer(
        reference,
        hypothesis,
        truth_transform=transforms,
        hypothesis_transform=transforms
    )

    print("WER:", error)

    # Store the error score
    error_scores.append(error)

    # Store the results
    results.append({'Clean Audio': os.path.basename(clean_audio),
                    'Noisy Audio': os.path.basename(noisy_audio),
                    'WER': error})
