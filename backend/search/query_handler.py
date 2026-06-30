import os
from typing import List, Dict, Any
from ingestion.embedding_generator import get_text_embedding
from db.chroma_client import search_segments

SCORE_THRESHOLD = 0.8


def merge_intervals(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes chunks belonging to ONE track.
    Merges chunks whose [start_time, end_time] intervals overlap or touch
    into single regions. Keeps the best (lowest) score within each region.
    """
    if not matches:
        return []

    matches = sorted(matches, key=lambda m: m["start_time"])
    merged = [dict(matches[0])]

    for current in matches[1:]:
        last = merged[-1]
        if current["start_time"] <= last["end_time"]:
            last["end_time"] = max(last["end_time"], current["end_time"])
            last["score"] = min(last["score"], current["score"])
        else:
            merged.append(dict(current))

    return merged


def group_by_track(matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}
    for match in matches:
        track = match["track_name"]
        if track not in grouped:
            grouped[track] = []
        grouped[track].append(match)
    return grouped


def search(query: str) -> List[Dict[str, Any]]:
    query_embedding = get_text_embedding(query)
    raw_results = search_segments(query_embedding, n_results=50)

    filtered = []
    for match in raw_results:
        if match["score"] > SCORE_THRESHOLD:
            continue
        if not os.path.exists(match["file_path"]):
            continue
        filtered.append(match)

    if not filtered:
        return []

    grouped = group_by_track(filtered)

    final_tracks = []
    for track_name, chunks in grouped.items():
        regions = merge_intervals(chunks)
        regions.sort(key=lambda r: r["score"])

        final_tracks.append({
            "track_name": track_name,
            "file_path":  chunks[0]["file_path"],
            "regions":    regions
        })

    final_tracks.sort(key=lambda t: t["regions"][0]["score"])

    return final_tracks