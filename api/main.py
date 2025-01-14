from fastapi import FastAPI
from typing import List
from models.Article import ArticlesInput
from models.Responses import *
from services.MyPipeine import MyPipeline
from pinecone import Pinecone
import time
from openai import OpenAI
import os
from decouple import config


# Initialize the NLP model (Sentiment Analysis pipeline as an example)
# nlp_pipeline = pipeline("sentiment-analysis")

# Define FastAPI app
app = FastAPI()

pipeline_instance = MyPipeline()

# Set your API key securely
openai_api_key = config("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Pinecone DB config
# TODO: De-couple from this file
key = config("PINECONE_KEY")
pc = Pinecone(api_key=key)
index_name = "nlp-project-2"
# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Initialize Pinecone Index
index = pc.Index(index_name)


@app.post("/get-embeddings/", response_model=EmbeddingResponse)
async def get_embeddings(input_data: ArticlesInput):
    # Generate embeddings for the provided list of texts
    embeddings = pipeline_instance.generate_embeddings(input_data.texts)
    
    # Return the embeddings in the correct format
    return EmbeddingResponse(embeddings=[embedding.tolist() for embedding in embeddings])

@app.post("/analyze-sentiment/", response_model=List[SentimentResponse])
async def analyze_sentiment(input_data: ArticlesInput):
    # Perform sentiment analysis for each text in the list
    sentiment_results = []
    for text in input_data.texts:
        sentiment, score = pipeline_instance.do_sentiment_analysis(text)
        sentiment_results.append(SentimentResponse(text=text, sentiment=sentiment, score=score))
    
    return sentiment_results


@app.post("/summarize/", response_model=List[SummarizeResponse])
async def summarize_texts(input_data: ArticlesInput):
    summaries = []
    for text in input_data.texts:
        summary = pipeline_instance.do_sentiment_analysis(text)
        summaries.append(SummarizeResponse(text=summary))
    
    return summaries


@app.post("/query-pinecone", response_model=List[PineconeQueryResponse])
async def query_pinecone(input_data: ArticlesInput):
    result = []

    text_embeddings = pipeline_instance.generate_embeddings(input_data.texts).tolist()

    for embedding in text_embeddings:
        query_result = index.query(
            namespace="news_namespace",
            vector=embedding,
            top_k=2,  # Number of similar results to retrieve
            include_metadata=True,  # Include metadata in the results
            include_values=True
        )


        for match in query_result["matches"]:
            content_summary = pipeline_instance.summarize_text(match['metadata']['content'])
            result.append(PineconeQueryResponse(id=match['id'], summary=content_summary, content=match['metadata']['content'], title=match['metadata']['title'], image_url=match['metadata']['image_url'], embedding=match['values'], sentiment="Positive"))

        # print(query_result)
        # result.append(PineconeQueryResponse(id=))
    
    return result


@app.post("/query-pinecone-v2", response_model=List[PineconeQueryResponse])
async def query_pinecone(input_data: ArticlesInput):
    result = []

    text_embeddings = []

    for text in input_data.texts:
        try:
            # Generate embedding
            response = client.embeddings.create(
                input=text.strip(),
                model="text-embedding-ada-002"
            )

            # Extract embedding vector
            embedding_vector = response.data[0].embedding
            text_embeddings.append(embedding_vector)
        except Exception as e:
            print(f"Error processing row {index + 1}: {e}")
            text_embeddings.append(None)


    for embedding in text_embeddings:
        query_result = index.query(
            namespace="news_namespace",
            vector=embedding,
            top_k=2,  # Number of similar results to retrieve
            include_metadata=True,  # Include metadata in the results
            include_values=True
        )


        for match in query_result["matches"]:
            sentiment = pipeline_instance.do_sentiment_analysis(match['metadata']['summary'])
            # embedding=match['values']
            result.append(PineconeQueryResponse(id=match['id'], summary=match['metadata']['summary'], content=match['metadata']['content'], title=match['metadata']['title'], image_url=match['metadata']['image_url'], embedding=[], sentiment=sentiment[0]))

        # print(query_result)
        # result.append(PineconeQueryResponse(id=))
    
    return result

