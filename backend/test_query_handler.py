from search.query_handler import search
import json

results = search("intense dramatic moment")
print(json.dumps(results, indent=2))