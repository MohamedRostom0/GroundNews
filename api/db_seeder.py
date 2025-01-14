import time
import json
import ast
from pinecone import Pinecone
from decouple import config


# Your Pinecone API key
key = config("PINECONE_KEY")



# Initialize Pinecone client
pc = Pinecone(api_key=key)

# Define index name and dimensions
index_name = "nlp-project-2"
embedding_dimension = 1536  # The dimensions should match the output of your embedding model


# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Your JSON data (reduced example)

with open("./filtered_news.json") as f:
  data = json.load(f)



# Initialize Pinecone Index
index = pc.Index(index_name)


# Prepare vectors for upsertion into Pinecone
vectors = []
for i, item in enumerate(data):
    if item.get('content'):  # Ensure we only process entries with valid content
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
                'content': item['content'],
                "url": item['url'],
            }
        })

# Check if vectors are empty
if not vectors:
    print("No valid content found in the data.")
else:
    # Upsert vectors into Pinecone
    index.upsert(vectors=vectors, namespace="news_namespace")

    # Check the stats of the index to confirm insertion
    print(index.describe_index_stats())