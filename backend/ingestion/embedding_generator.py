import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
from typing import List

print("Loading CLAP model... (this may take a moment)")
model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model.eval()
print("CLAP model loaded successfully")


def normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


def get_audio_embedding(audio_chunk: np.ndarray, sample_rate: int) -> List[float]:
    inputs = processor(
        audio=audio_chunk.tolist(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    with torch.no_grad():
        output = model.get_audio_features(**inputs)

    # extract tensor from output object
    if hasattr(output, 'pooler_output'):
        tensor = output.pooler_output
    elif hasattr(output, 'last_hidden_state'):
        tensor = output.last_hidden_state.mean(dim=1)
    else:
        tensor = output  # already a tensor

    tensor = normalize(tensor)
    return tensor.squeeze().tolist()


def get_text_embedding(text: str) -> List[float]:
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        output = model.get_text_features(**inputs)

    if hasattr(output, 'pooler_output'):
        tensor = output.pooler_output
    elif hasattr(output, 'last_hidden_state'):
        tensor = output.last_hidden_state.mean(dim=1)
    else:
        tensor = output  # already a tensor

    tensor = normalize(tensor)
    return tensor.squeeze().tolist()