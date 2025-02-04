import time
import json
import ast
from pinecone import Pinecone
from decouple import config
from services.MyPipeine import MyPipeline


# Your Pinecone API key
key = config("PINECONE_KEY")
pipeline_instance = MyPipeline()


# Initialize Pinecone client
pc = Pinecone(api_key=key)

# Define index name and dimensions
index_name = "nlp-project-final"
embedding_dimension = 1536  # The dimensions should match the output of your embedding model


# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Your JSON data (reduced example)

with open("./final_data.json") as f:
  data = json.load(f)



# Initialize Pinecone Index
index = pc.Index(index_name)


# Prepare vectors for upsertion into Pinecone
vectors = []
for i, item in enumerate(data):
    if item.get('content'):  # Ensure we only process entries with valid content
        sentiment = pipeline_instance.do_sentiment_analysis(item['content'])
        with open(f"./data/{str(item['id'])}.txt", "w") as file:
            file.write(item['content'])

        vectors.append({
            "id": str(item['id']),
            "values": ast.literal_eval(item['embeddings']),
            "metadata": {
                "title": item['title'],
                "query": item.get('query', ""),
                "domain": item.get('domain', ""),
                "description": item['description'],
                "publishedAt": item['publishedAt'],
                "summary": item['summary'],
                "source": item['source'],
                "image_url": item.get('image_url', ""),
                # 'content': item['content'],
                "url": item['url'],
                "sentiment": sentiment[0]
            }
        })


# Define the maximum size limit (4 MB in bytes)
MAX_SIZE = 4 * 1024 * 1024

# Function to split vectors into smaller chunks
def chunk_and_insert_vectors(vectors, chunk_size=1):
    total_vectors = len(vectors)
    for i in range(0, total_vectors, chunk_size):
        # Create a chunk from the vector list
        chunk = vectors[i:i + chunk_size]
        
        # Upsert the chunk into Pinecone
        index.upsert(vectors=chunk, namespace="news_namespace")


# Split the vectors into chunks
chunk_and_insert_vectors(vectors)


# # Upsert each chunk separately
# for chunk in vector_chunks:
#     index.upsert(vectors=chunk, namespace="news_namespace")