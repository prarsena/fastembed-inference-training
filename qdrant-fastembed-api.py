import os
import json
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from dotenv import load_dotenv
from huggingface_hub import login, whoami

##  1. Setup - Load env vars and Connect to Qdrant

load_dotenv()
#print(whoami())
# client = QdrantClient(host="localhost", port=6333)
client = QdrantClient(":memory:")  # Qdrant is running from RAM.


##  2. Ingestion - Your documents

docs = [
    "mall (n.) - 1737, \"shaded walk serving as a promenade,\" generalized from The Mall, name of a broad, tree-lined promenade in St. James's Park, London (so called from 1670s, earlier Maill, 1640s), which was so called because it formerly was an open alley that was used to play pall-mall.",
    "Easter (n.) - Bede writes that Anglo-Saxon Christians adopted her name for their Mass of Christ's resurrection. Almost all neighboring languages use a variant of Latin Pascha to name this holiday.",
    "take (v.) - Middle English taken, from late Old English tacan \"to grip, seize by force, lay hold of,\" from a Scandinavian source (such as Old Norse taka \"take, grasp, lay hold,\" past tense tok, past participle tekinn; also compare Swedish ta, past participle tagit).",
    "butterfly(n.) - common name of any lepidopterous insect active in daylight, Old English buttorfleoge, evidently butter (n.) + fly (n.), but the name is of obscure signification. Perhaps based on the old notion that the insects (or, according to Grimm, witches disguised as butterflies) consume butter or milk that is left uncovered."
]

metadata = [
    {"source": "https://www.etymonline.com/word/mall"},
    {"source": "https://www.etymonline.com/word/Easter"},
    {"source": "https://www.etymonline.com/word/take"},
    {"source": "https://www.etymonline.com/word/butterfly"},
]

ids = [42, 2, 66, 23]
collection_name="eytomology_kb"
model_name = "BAAI/bge-small-en"


## Create Qdrant collection

client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), 
        distance=models.Distance.COSINE
    ),  # size and distance are model dependent
)


## Upsert documents to the collection

metadata_with_docs = [
    {"document": doc, "source": meta["source"]} for doc, meta in zip(docs, metadata)
]

client.upload_collection(
    collection_name=collection_name,
    vectors=[models.Document(text=doc, model=model_name) for doc in docs],
    payload=metadata_with_docs,
    ids=ids,
)

# 3. Interaction - User asks a question, Qdrant searches for most relevant context, and returns it:
user_query = "What insect's name derives from the belief they consume uncovered butter?"

search_result = client.query_points(
    collection_name=collection_name,
    query=models.Document(text=user_query, model=model_name)
).points

results_as_dicts = [point.model_dump() for point in search_result]
# print(json.dumps(results_as_dicts, indent=4, sort_keys=True))

retrieved_context_document = results_as_dicts[0]["payload"]["document"]
retrieved_context_source   = results_as_dicts[0]["payload"]["source"]
print(f"Retrieved Context: {retrieved_context_document}")

