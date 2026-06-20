import librosa
import numpy as np
from typing import List, Tuple

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file and returns the audio signal and sample rate.
    audio signal = the actual sound data as numbers
    sample rate  = how many numbers represent 1 second (usually 44100)
    """
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return audio, sample_rate


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_size: int = 3,
    step_size: int = 1
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Slices audio into overlapping chunks.
    Returns a list of (chunk, start_time, end_time) tuples.
    """
    window_samples = window_size * sample_rate  # 3 seconds worth of numbers
    step_samples   = step_size * sample_rate    # 1 second worth of numbers

    chunks = []
    start = 0

    while start + window_samples <= len(audio):
        end        = start + window_samples
        chunk      = audio[start:end]
        start_time = start / sample_rate        # convert back to seconds
        end_time   = end / sample_rate

        chunks.append((chunk, start_time, end_time))
        start += step_samples                   # slide forward by 1 second

    return chunks