from typing import Annotated
from enum import Enum
import httpx
from fastapi import FastAPI, Query
from sentence_transformers import CrossEncoder, SentenceTransformer
import os


## Definitions
## ------------------------------------------------------------------
CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123/")
N_RERANK = int(os.environ.get("N_RERANK", 100))
N_OUTPUT = int(os.environ.get("N_OUTPUT", 5))

class SearchIndexLabel(str, Enum):
    video_transcription = "video_transcription"


## Initialization
## ------------------------------------------------------------------
app = FastAPI()
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


## Endpoints
## ------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return "OK"

@app.get("/api/v1/{search_index}/search")
async def root(search_index: SearchIndexLabel, query: Annotated[str, Query(max_length=100)], rerank: bool = True):
    query_embedding = embedder.encode(query, convert_to_tensor=True).tolist()

    n_rows = N_OUTPUT if not rerank else N_RERANK;
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{CLICKHOUSE_URL}?query=SELECT text, doc_id, start, cosineDistance(dense, {query_embedding}) AS distance FROM {search_index.value} ORDER BY distance LIMIT {n_rows};")

    if not rerank:
        return list(_postprocess_ir_outputs(res.text))

    inputs, meta = list(zip(*list(_preprocess_rerank_inputs(query, res.text))))
    scores = cross_encoder.predict(inputs)
    outputs = sorted(_postprocess_rerank_outputs(scores, meta), key=lambda elem: elem["score"], reverse=True)
    outputs = outputs[:N_OUTPUT]

    return outputs


## Helper functions
## ------------------------------------------------------------------
def _make_video_uri(doc_id, start):
    return f"https://netology-group.services/webinar-foxford/dispatcher/api/v1/redirs/tenants/foxford/apps/webinar?embedded_origin=https://foxford.ru&scope={doc_id}&t={start}"

def _postprocess_ir_outputs(outputs):
    for line in outputs.strip().split('\n'):
        text, doc_id, start, distance = line.split('\t')
        yield {
            "text": text,
            "uri": _make_video_uri(doc_id, start),
            "distance": float(distance),
        }

def _preprocess_rerank_inputs(query, outputs):
    for line in outputs.strip().split('\n'):
        text, doc_id, start, distance = line.split('\t')
        meta = {
            "text": text,
            "uri": _make_video_uri(doc_id, start),
            "distance": float(distance),
        }
        data = [query, text]
        yield data, meta

def _postprocess_rerank_outputs(scores, meta):
    for n, score in enumerate(scores.tolist()):
        yield {**meta[n], "score": score}
