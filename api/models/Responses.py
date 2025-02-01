from pydantic import BaseModel
from typing import List

# Response model for embeddings
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]  # List of embeddings (each embedding is a list of floats)

# Response model for sentiment analysis
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    score: float


class SummarizeResponse(BaseModel):
    summary: str


class PineconeQueryResponse(BaseModel):
    id: str
    title: str
    content: str
    summary: str
    image_url: str
    embedding: List[float]
    sentiment: str

class PartyPCResponse(BaseModel):
    partyName: str
    fullText: str
    chunkText: str