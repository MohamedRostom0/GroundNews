from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class MyPipeline:
    def __init__(self) -> None:
        self.sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


    def generate_embeddings(self, texts):
        return self.embeddings_model.encode(texts)

    def do_sentiment_analysis(self, text: str):
        print(text)
        result = self.sentiment_model(text[:512])
        sentiment = result[0]['label']
        score = result[0]['score']

        return sentiment, score
    
    def cluster_embeddings(self, k, embeddings):
        kmeans = KMeans(n_clusters=k, random_state=42)
        return kmeans.fit_predict(embeddings)
    

    
    def summarize_text(self, text):
        def chunk_text(text, max_chunk_size=500):
            words = text.split()
            for i in range(0, len(words), max_chunk_size):
                yield " ".join(words[i:i + max_chunk_size])

        summaries = []
        for chunk in chunk_text(text, max_chunk_size=500):
            summary = self.summarizer(chunk, max_length=350, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])

            # Combine all chunk summaries
            return " ".join(summaries)

