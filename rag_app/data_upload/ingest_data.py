# File: open_party_rag/rag_app/data_upload/ingest_data.py
import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from pinecone import Pinecone, ServerlessSpec
from input import INPUT_FOLDER
from processed import move_file

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-index"

def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        sample_embedding = client.embeddings.create(input="sample", model="text-embedding-ada-002")
        dimension = len(sample_embedding.data[0].embedding)
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENV
            )
        )
    return pc.Index(INDEX_NAME)

def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def process_file(file_path, index):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(data)
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vector = {
            "id": f"{os.path.basename(file_path)}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": file_path,
                "page": i
            }
        }
        vectors.append(vector)
    index.upsert(vectors=vectors)

def main():
    index = initialize_pinecone()
    if not os.path.exists(INPUT_FOLDER):
        raise Exception("Input folder does not exist: " + INPUT_FOLDER)
    txt_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    for file_path in txt_files:
        try:
            process_file(file_path, index)
            move_file(file_path)
        except Exception as e:
            raise Exception(f"Failed to process file {file_path}: {str(e)}")
    
if __name__ == "__main__":
    main()
