import os
from ingestion.embedding_generator import get_text_embedding
from db.chroma_client import search_segments

SCORE_THRESHOLD = 0.8  # based on scores we saw (0.72-0.79)

def search(query: str):
    print(f"\nSearching for: '{query}'")

    query_embedding = get_text_embedding(query)
    raw_results = search_segments(query_embedding, n_results=10)

    final_results = []
    for match in raw_results:
        if match["score"] > SCORE_THRESHOLD:
            continue
        if not os.path.exists(match["file_path"]):
            continue
        final_results.append(match)
        if len(final_results) == 5:
            break

    if not final_results:
        print("No matching audio found")
        return

    print(f"Found {len(final_results)} matches:\n")
    for r in final_results:
        mins = int(r["start_time"] // 60)
        secs = int(r["start_time"] % 60)
        print(f"  Track:     {r['track_name']}")
        print(f"  Timestamp: {mins:02d}:{secs:02d}")
        print(f"  Score:     {r['score']:.3f}")
        print()

search("calm peaceful music")
search("intense dramatic moment")
search("soft gentle melody")
search("building tension")