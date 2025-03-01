# File: open_party_rag/rag_app/views.py
import os
import json
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
INDEX_NAME = "rag-index"
if INDEX_NAME not in pc.list_indexes():
    sample_embedding = openai.embeddings.create(input="sample", model="text-embedding-ada-002")
    dimension = len(sample_embedding.data[0].embedding)
    region = os.getenv("PINECONE_ENV")
    cloud = "gcp" if "gcp" in region else "aws"
    pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud=cloud,
            region=region
        )
    )
index = pc.Index(INDEX_NAME)

def get_embedding(text):
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

@csrf_exempt
def query_rag(request):
    if request.method == "POST":
        body = json.loads(request.body)
        user_query = body.get("query")
        query_embedding = get_embedding(user_query)
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        context = ""
        for item in results["matches"]:
            metadata = item["metadata"]
            context += "Source: " + metadata.get("source", "") + " Page: " + str(metadata.get("page", "")) + " Text: " + metadata.get("text", "") + "\n"
        prompt = "Context: " + context + "\nUser Query: " + user_query + "\nAnswer:"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=512
        )
        answer = response["choices"][0]["message"]["content"]
        return JsonResponse({"answer": answer})
    return JsonResponse({"error": "Invalid request method"}, status=400)
