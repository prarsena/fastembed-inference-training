from typing import List
import numpy as np
from fastembed import TextEmbedding
import englishLanguageData

documents = englishLanguageData.word_entries
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
definitions = [item["definition"] for item in documents]
vector_embeddings = list(embedding_model.embed(definitions))

print(f"Model dimensions: {len(vector_embeddings[0])}")
#print("Embeddings for item #0:\n", vector_embeddings[0])

user_query = input("Your Query: ")

# 4. Vectorize the Query (must use the same model!)
# We wrap it in a list because .embed() expects an iterable
query_embedding = list(embedding_model.embed([user_query]))[0]

# 5. Calculate Similarity (The "Search" step)
# We use np.dot to see how much the query "overlaps" with each document
scores = [np.dot(query_embedding, doc_emb) for doc_emb in vector_embeddings]

# 6. Get the result
best_index = np.argmax(scores) # Find the index of the highest score
highest_score = scores[best_index]

print(f"\nQuery: {user_query}")
#print(f"Most relevant document: {documents[best_index]}")
print(f"Most relevant document: {documents[best_index]['word']} - {documents[best_index]['definition']}")
print(f"Similarity Score: {highest_score:.4f}")