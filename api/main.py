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
import json
import os
# from langchain_community.llms import OpenAI
import langchain as lc
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA



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
index_name = "nlp-project-final"
parties_index_name = "german-parties-2"
# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)
while not pc.describe_index(parties_index_name).status['ready']:
    time.sleep(1)

# Initialize Pinecone Index
index = pc.Index(index_name)
parties_index = pc.Index(parties_index_name)


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
            with open(f"./data/{match['id']}.txt", "r") as file:
                content = file.read()

            content_summary = pipeline_instance.summarize_text(content)


            result.append(PineconeQueryResponse(id=match['id'], summary=content_summary, content=content, title=match['metadata']['title'], image_url=match['metadata']['image_url'], embedding=match['values'], sentiment="Positive"))

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

            with open(f"./data/{match['id']}.txt", "r") as file:
                content = file.read()
            # embedding=match['values']
            result.append(PineconeQueryResponse(id=match['id'], summary=match['metadata']['summary'], content=content, title=match['metadata']['title'], image_url=match['metadata']['image_url'], embedding=[], sentiment=sentiment[0]))

        # print(query_result)
        # result.append(PineconeQueryResponse(id=))
    
    return result


@app.post("/vote")
async def vote(input_data: str):
    with open('./data/likes.json', "r") as file:
        likedIds = json.load(file)

    liked_articles = index.fetch(namespace='news_namespace', ids=likedIds)
    liked_summaries = []
    for id in likedIds:
        # print(liked_articles['vectors'][id]['metadata']['summary'])
        liked_summaries.append(liked_articles['vectors'][id]['metadata']['summary'])

    # with open('./logs.json', "w") as file:
    #     json.dump(liked_articles_dict, file, indent=4)


    clarification_template = PromptTemplate(
        input_variables=["query"],
        template="Please clarify this query and make it informative to be used in RAG: {query}"
    )
    
    qa_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question at the end. 
        Context: {context}
        Question: {question}"""
    )
    
    # Define the chains
    clarification_chain = LLMChain(llm=lc.OpenAI(temperature=0, openai_api_key=openai_api_key), prompt=clarification_template)
    qa_chain = LLMChain(llm=lc.OpenAI(temperature=0, openai_api_key=openai_api_key), prompt=qa_template)
    
    
    # Clarify the user's query
    clarified_query = await clarification_chain.arun(query=input_data)
    
    # Perform similarity search in the vector database
    # docs = db.similarity_search(clarified_query, k=3)  # Replace `db` with your database object
    embedding_vector = None
    try:
        # Generate embedding
        response = client.embeddings.create(
            input=clarified_query.strip(),
            model="text-embedding-ada-002"
        )

        # Extract embedding vector
        embedding_vector = response.data[0].embedding
    except Exception as e:
        print(f"Error processing row: {e}")


    res = parties_index.query(
            namespace="parties_namespace",
            vector=embedding_vector,
            top_k=5,  # Number of similar results to retrieve
            include_metadata=True,  # Include metadata in the results
            include_values=False
        )
    docs = []
    for match in res["matches"]:
        docs.append(match['metadata']['chunkTxt'])

    
    # Combine the relevant context
    context = "\n".join([doc for doc in docs])
    
    # Formulate the question based on user input and context
    question = (
        "Here are the three possible parties' policies for me to vote on. "
        f"My question is: which one is the best for me? My opinion is {input_data}. "
        "If two parties are equally suitable, mention both of them."
    )
    
    # Generate the answer
    answer = await qa_chain.arun(context=context, question=question)
    
    return answer


@app.post("/vote2")
async def vote2(input_data: str):
    with open('./data/likes.json', "r") as file:
        likedIds = json.load(file)

    liked_articles = index.fetch(namespace='news_namespace', ids=likedIds)
    liked_summaries = [
        liked_articles['vectors'][id]['metadata']['summary'] for id in likedIds
    ]

    clarification_template = PromptTemplate(
        input_variables=["query"],
        template="Please clarify this query and make it informative to be used in RAG: {query}"
    )

    qa_template = PromptTemplate(
        input_variables=["context", "liked_summaries", "question"],
        template="""
        Use the following context and previously liked articles to answer the question:
        
        ### Previously Liked Articles:
        {liked_summaries}
        
        ### Context:
        {context}
        
        ### Question:
        {question}
        """
    )

    # Define the chains
    clarification_chain = LLMChain(llm=lc.OpenAI(temperature=0, openai_api_key=openai_api_key), prompt=clarification_template)
    qa_chain = LLMChain(llm=lc.OpenAI(temperature=0, openai_api_key=openai_api_key), prompt=qa_template)

    # Clarify the user's query
    clarified_query = await clarification_chain.arun(query=input_data)

    # Generate embedding for similarity search
    embedding_vector = None
    try:
        response = client.embeddings.create(
            input=clarified_query.strip(),
            model="text-embedding-ada-002"
        )
        embedding_vector = response.data[0].embedding
    except Exception as e:
        print(f"Error processing row: {e}")

    # Perform similarity search
    res = parties_index.query(
        namespace="parties_namespace",
        vector=embedding_vector,
        top_k=5,
        include_metadata=True,
        include_values=False
    )

    docs = [match['metadata']['chunkTxt'] for match in res["matches"]]

    # Combine the relevant context
    context = "\n".join(docs)

    # Formulate the question with liked_summaries
    question = (
        "Here are the three possible parties' policies for me to vote on. "
        f"My question is: which one is the best for me? My opinion is {input_data}. "
        "If two parties are equally suitable, mention both of them."
    )

    # Generate the answer, including liked_summaries in the prompt
    answer = await qa_chain.arun(
        context=context, 
        liked_summaries="\n".join(liked_summaries),
        question=question
    )

    return answer



@app.post("/like")
async def like_article(postId: str):
    likes = []
    with open('./data/likes.json', "r") as file:
        likes = json.load(file)

    if postId not in likes:
        likes.append(postId)
    else:
        likes.remove(postId)


    with open('./data/likes.json', "w") as file:
        json.dump(likes, file)

    return likes


@app.delete("/purge/likes")
async def like_article():
    likes = []

    with open('./data/likes.json', "w") as file:
        json.dump(likes, file)

    return likes
