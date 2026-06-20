import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
from typing import List

print("Loading CLAP model... (this may take a moment)")
model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model.eval()
print("CLAP model loaded successfully")


def get_audio_embedding(audio_chunk: np.ndarray, sample_rate: int) -> List[float]:
    inputs = processor(
        audio=audio_chunk.tolist(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.audio_model(**inputs)

    # pooler_output is the final summarized embedding
    embedding = output.pooler_output.squeeze()
    return embedding.tolist()


def get_text_embedding(text: str) -> List[float]:
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        output = model.text_model(**inputs)

    embedding = output.pooler_output.squeeze()
    return embedding.tolist()