import os
import uuid
import hashlib
from ingestion.audio_chunker import load_audio, chunk_audio
from ingestion.embedding_generator import get_audio_embedding
from db.chroma_client import store_segment, is_already_indexed, delete_segments_by_track


def get_file_hash(file_path: str) -> str:
    """
    Generates a unique SHA256 fingerprint of the file content
    If the file changes even by 1 byte, the hash changes completely
    """
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def index_audio_file(file_path: str):
    track_name = os.path.basename(file_path)
    file_hash  = get_file_hash(file_path)

    # check if this exact file is already indexed
    if is_already_indexed(file_path, file_hash):
        print(f"Skipping {track_name} - already indexed")
        return

    # file exists but with different hash = file was modified
    # delete old segments and re-index fresh
    delete_segments_by_track(file_path)

    print(f"Indexing: {track_name}")

    audio, sr = load_audio(file_path)
    print(f"Loaded: {len(audio)/sr:.1f} seconds of audio")

    chunks = chunk_audio(audio, sr)
    print(f"Created {len(chunks)} chunks")

    for i, (chunk, start_time, end_time) in enumerate(chunks):
        segment_id = str(uuid.uuid4())
        embedding  = get_audio_embedding(chunk, sr)

        store_segment(
            segment_id=segment_id,
            embedding=embedding,
            track_name=track_name,
            file_path=file_path,
            file_hash=file_hash,
            start_time=start_time,
            end_time=end_time
        )

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(chunks)} chunks indexed")

    print(f"Done: {track_name}")


def index_folder(folder_path: str):
    supported = (".wav", ".mp3")
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(supported)
    ]

    if not files:
        print("No audio files found in folder")
        return

    print(f"Found {len(files)} audio files")
    for file_path in files:
        index_audio_file(file_path)

    print("All files indexed successfully")