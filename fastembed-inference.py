from typing import List
import numpy as np
from fastembed import TextEmbedding

documents: List[str] = [
    "1749 Diary: Today we prepared a Tansy Cake, heavy on the cream and eggs.",
    "Cookbook: To make a modern chocolate lava cake, preheat your oven to 450F."
]

embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

embeddings_generator = embedding_model.embed(documents)
embeddings_list = list(embeddings_generator)
len(embeddings_list[0])  

# print("Embeddings:\n", embeddings_list)

# 3. Add the User Query
user_query = "What kind of desserts were made in the 18th century?"

# 4. Vectorize the Query (must use the same model!)
# We wrap it in a list because .embed() expects an iterable
query_embedding = list(embedding_model.embed([user_query]))[0]

# 5. Calculate Similarity (The "Search" step)
# We use np.dot to see how much the query "overlaps" with each document
scores = [np.dot(query_embedding, doc_emb) for doc_emb in embeddings_list]

# 6. Get the result
best_index = np.argmax(scores) # Find the index of the highest score
highest_score = scores[best_index]

print(f"\nQuery: {user_query}")
print(f"Most relevant document: {documents[best_index]}")
print(f"Similarity Score: {highest_score:.4f}")