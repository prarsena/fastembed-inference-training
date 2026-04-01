# Building a Semantic Search Engine with FastEmbed and Qdrant

Unlike a traditional "Ctrl+F" search that looks for exact character matches, this script understands the meaning of words. 

If you search for "a large body of water," it can find "Ocean" even if the word "water" isn't in the definition.


## What's Happening in this Script?

This code has three architectural phases:

- The Brain (Embedding): When you pass model_name = "BAAI/bge-small-en-v1.5", you are loading a pre-trained model that has "read" a massive chunk of the internet. It turns text into a vector (a list of numbers). In your code, it turns your dictionary definitions into 384-dimensional coordinates.

- The Library (Vector DB): Qdrant is acting as your high-speed librarian. By setting it to :memory:, you’re telling it to store all those coordinates in your RAM rather than on your hard drive. It organizes these numbers so it can calculate the "distance" between two ideas instantly.

- The Retrieval (Cosine Similarity): When you type a query, the model turns your query into a vector and asks Qdrant, "Which point in the library is physically closest to this query's coordinate?" The "Cosine" distance is just a fancy way of measuring the angle between two vectors.


## Install and Run

```python
pip install -r req.txt
```

Uses the hf model: https://huggingface.co/BAAI/bge-small-en-v1.5
