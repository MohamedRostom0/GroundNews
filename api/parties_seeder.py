from pinecone import Pinecone
from decouple import config
import time
import ast
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI

nltk.download('punkt_tab')

# Keys
key = config("PINECONE_KEY")
openai_api_key = config("OPENAI_API_KEY")

pc = Pinecone(api_key=key)
client = OpenAI(api_key=openai_api_key)

index_name = "german-parties-2"
embedding_dimension = 1536

# Wait for the index to be ready
while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

# Initialize Pinecone Index
index = pc.Index(index_name)


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    
    return text


def split_text_into_chunks(text, max_length=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def generate_embeddings_openai(texts):
    text_embeddings = []

    for text in texts:
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
            text_embeddings.append(None)

        print(len(text_embeddings))
    
    return text_embeddings


# print(generate_embeddings_openai(["I love Germnay"]))
# party_txt = extract_text_from_pdf("./parties/Alliance.pdf")
# chunks = split_text_into_chunks(party_txt)
# print(len(chunks))
# print()

def get_parties_texts(partyNames):
    # texts = []
    partyNameToText = {}
    for pName in partyNames:
        party_txt = extract_text_from_txt(f"./parties/{pName}")
        # texts.append(party_txt)
        partyNameToText[pName.split(".")[0]] = party_txt
    
    print("Finished get_parties_texts")
    return partyNameToText

partyNameToTextMap = get_parties_texts(["AFD.txt", "Alliance.txt", "CDU-CSU.txt", "FDP.txt", "SPD.txt", "TheLeft.txt", "VOLT.txt"])
# print(partyNameToTextMap)

vectors = []
for partyName in partyNameToTextMap:
    partyTxt = partyNameToTextMap[partyName]
    chunks = split_text_into_chunks(partyTxt)
    chunkEmbeddings = generate_embeddings_openai(chunks)

    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": f"{partyName}-chunk{i}",
            "values": chunkEmbeddings[i],
            "metadata": {
                "party": partyName,
                # "fullText": partyTxt,
                "chunkTxt": chunk
            }
        })

# Function to split vectors into smaller chunks
def chunk_and_insert_vectors(vectors, chunk_size=1):
    total_vectors = len(vectors)
    for i in range(0, total_vectors, chunk_size):
        # Create a chunk from the vector list
        chunk = vectors[i:i + chunk_size]
        
        # Upsert the chunk into Pinecone
        index.upsert(vectors=chunk, namespace="parties_namespace")


# Split the vectors into chunks
chunk_and_insert_vectors(vectors, 2)
