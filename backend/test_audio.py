from ingestion.audio_chunker import load_audio, chunk_audio
from ingestion.embedding_generator import get_audio_embedding, get_text_embedding

# Load and chunk audio
audio, sr = load_audio(r'C:/Users/rajpr/Desktop/CODE/sonarc/backend/assets/Lenguas - Elephant Heart.mp3')
chunks = chunk_audio(audio, sr)

# Test audio embedding
first_chunk, start, end = chunks[0]
embedding = get_audio_embedding(first_chunk, sr)
# import torch
# from transformers import ClapModel, ClapProcessor

# model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
# processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

# inputs = processor(
#     audio=first_chunk.tolist(),
#     sampling_rate=sr,
#     return_tensors="pt"
# )

# with torch.no_grad():
#     output = model.get_audio_features(**inputs)

# print(f"Output type: {type(output)}")
# print(f"Output keys if dict: {output.keys() if hasattr(output, 'keys') else 'not a dict'}")
# print(f"Output attrs: {[attr for attr in dir(output) if not attr.startswith('_')]}")

# Test text embedding
text_emb = get_text_embedding("intense dramatic music")

# Print summary only - no giant float lists
print("="*40)
print(f"Total chunks: {len(chunks)}")
print(f"Audio embedding length: {len(embedding)}")
print(f"Text embedding length: {len(text_emb)}")
print(f"Same size: {len(embedding) == len(text_emb)}")
print(f"Audio first 3 values: {embedding[:3]}")
print(f"Text first 3 values: {text_emb[:3]}")
print("="*40)