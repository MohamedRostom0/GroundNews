import json
import numpy as np
import ast
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from services.MyPipeine import MyPipeline
from sklearn.manifold import TSNE


# # Load the embeddings from JSON
# with open("all_embeddings.json", "r") as f:
#     all_embeddings = json.load(f)

pipeline_instance = MyPipeline()

with open("./data_31Jan.json") as f:
  data = json.load(f)

all_embeddings = []
all_titles = []
all_summaries = []
all_content = []
all_published_at = []
all_domain = []
all_queries = []
all_descriptions = []
all_image_urls = []
all_ids = []

for i, item in enumerate(data):
    all_embeddings.append(ast.literal_eval(item['embeddings']))
    all_titles.append(item['title'])
    all_summaries.append(item['summary'])
    all_content.append(item['content'])
    all_published_at.append(item['publishedAt'])
    all_domain.append(item.get('domain', ""))
    all_queries.append(item.get('query', ""))
    all_image_urls.append(item.get('image_url', ""))
    all_descriptions.append(item['description'])
    all_ids.append(item['id'])




# Convert to NumPy array
l = np.array(all_embeddings)

# Check shape of embeddings (ensure it's 2D)
print(f"Shape of embeddings: {l.shape}")
# pca = PCA(n_components=10)
# reduced_embeddings = pca.fit_transform(l)

l_normalized = StandardScaler().fit_transform(l)


# DBSCAN clustering

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(l_normalized)

# Plot the data


dbscan = DBSCAN(eps=1.5, min_samples=2)  # Adjust eps as needed
labels = dbscan.fit_predict(embeddings_2d)


clusters = {}
for i,label in enumerate(labels):
   if str(label) not in clusters:
    clusters[str(label)] = []
   
   sentiment = pipeline_instance.do_sentiment_analysis(all_summaries[i])
   obj = {'id': all_ids[i] ,'title': all_titles[i], 'summary': all_summaries[i], 'content': all_content[i], 'publishedAt': all_published_at[i], 'domain': all_domain[i], 'query': all_queries[i], 'image_url': all_image_urls[i], 'description': all_descriptions[i], 'sentiment': sentiment[0],"label":str(label)}
   clusters[str(label)].append(obj)

# Write the dictionary to a JSON file
with open("clusters.json", "w") as f:
    json.dump(clusters, f, indent=4) 

        
# print("=========")
# print(f"Max label (number of clusters): {max(labels)}")


# print(labels)