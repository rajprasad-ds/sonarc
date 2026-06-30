import chromadb
from typing import List, Dict, Any

client = chromadb.PersistentClient(path="./chroma_store")

collection = client.get_or_create_collection(
    name="audio_segments",
    metadata={"hnsw:space": "cosine"}
)


def store_segment(
    segment_id: str,
    embedding: List[float],
    track_name: str,
    file_path: str,
    file_hash: str,
    start_time: float,
    end_time: float
):
    collection.add(
        ids=[segment_id],
        embeddings=[embedding],
        metadatas=[{
            "track_name": track_name,
            "file_path":  file_path,
            "file_hash":  file_hash,
            "start_time": start_time,
            "end_time":   end_time
        }]
    )


def is_already_indexed(file_path: str, file_hash: str) -> bool:
    """
    Returns True if this exact file (same path AND same hash) 
    is already in ChromaDB
    """
    results = collection.get(
        where={
            "$and": [
                {"file_path": {"$eq": file_path}},
                {"file_hash": {"$eq": file_hash}}
            ]
        },
        limit=1
    )
    return len(results["ids"]) > 0


def delete_segments_by_track(file_path: str):
    """
    Deletes all segments for a file — used when file is 
    replaced with a newer version
    """
    results = collection.get(
        where={"file_path": {"$eq": file_path}}
    )
    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"Deleted {len(results['ids'])} old segments for {file_path}")


def search_segments(
    query_embedding: List[float],
    n_results: int = 10
) -> List[Dict[str, Any]]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    raw_matches = []
    for i in range(len(results["ids"][0])):
        raw_matches.append({
            "segment_id": results["ids"][0][i],
            "track_name": results["metadatas"][0][i]["track_name"],
            "file_path":  results["metadatas"][0][i]["file_path"],
            "start_time": results["metadatas"][0][i]["start_time"],
            "end_time":   results["metadatas"][0][i]["end_time"],
            "score":      results["distances"][0][i]
        })

    return raw_matches