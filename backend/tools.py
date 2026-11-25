"""Small allowlisted tool surface; no arbitrary functions or shell access."""
import math


def search_documents(store, query):
    sources = []
    for hit in store.search(query, 4):
        if math.isfinite(hit['distance']) and hit['distance'] <= 0.65:
            sources.append({**hit, 'text': hit['text'][:1200], 'citation_id': len(sources) + 1})
    return sources


def save_approved_plan(db, plan_id):
    return db.approve_plan(plan_id)
