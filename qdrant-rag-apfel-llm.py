import json
import englishLanguageData                                  ## Assumes you have englishLanguageData.py
from openai import OpenAI
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from huggingface_hub import login, whoami

##  1. Setup 

load_dotenv()                                                   # Load env vars

## In a terminal, run `apfel --serve`: 
llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="apfel")
client = QdrantClient(":memory:")                               # Qdrant is running from RAM.
collection_name="eytomology_kb"
model_name = "BAAI/bge-small-en-v1.5"                           # Specify HF model
word_entries = englishLanguageData.word_entries                 # Import your English dictionary. 

##  2. Ingestion - Send documents to db

client.create_collection(                                       # Create Qdrant collection
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), 
        distance=models.Distance.COSINE
    ),                                                          # size and distance are model dependent
)

client.upload_collection(
    collection_name=collection_name,
    vectors=[
        models.Document(text=item["definition"], model=model_name) 
        for item in word_entries
    ],
    payload=[
        {k: v for k, v in item.items() if k != "id"}            # All keys except ID go in payload
        for item in word_entries
    ],
    ids=[item["id"] for item in word_entries],
)


print(f"The model dimension is: {client.get_embedding_size(model_name)}")

# 3. Interaction - Qdrant searches for most relevant context

user_query = input("Your query: ")
#print (f"User Query: {user_query}")

search_result = client.query_points(
    collection_name=collection_name,
    query=models.Document(text=user_query, model=model_name)
).points

results_as_dicts = [point.model_dump() for point in search_result]
#print(json.dumps(results_as_dicts, indent=4, sort_keys=True))

retrieved_contexts = [res for res in results_as_dicts[:3]]
print(retrieved_contexts)
retrieved_context    = results_as_dicts[0]["payload"]
#print(retrieved_context)
retrieved_word       = results_as_dicts[0]["payload"]["word"]
retrieved_definition = results_as_dicts[0]["payload"]["definition"]
print(f"RAG Context Retrieved: {retrieved_word} - {retrieved_definition}")

## set up LLM integration

def generate_answer(user_query, retrieved_context):
    prompt = f"""
    Use the following context to answer the question.
    Context: {retrieved_context}
    Question: {user_query}
    """
    
    response = llm_client.chat.completions.create(
        model="apple-foundationmodel",
        messages=[
            {"role": "system", "content": "Evaluate the retrieved word definitions and scores, and return your preferred option of the three."},
            {"role": "user", "content": prompt}
        ]
        #stream=True # Apfel supports native streaming
    )

    
    print(response.choices[0].message.content)

    # for chunk in response:
    #     if chunk.choices[0].delta.content:
    #         print(chunk.choices[0].delta.content, end="", flush=True)


generate_answer(user_query, retrieved_contexts)