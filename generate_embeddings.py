from sentence_transformers import SentenceTransformer
import os

def generate_embeddings(texts, model_name):
    model = SentenceTransformer(model_name)
    return model.encode(texts)

if __name__ == '__main__':
    sample_texts = ["This is a sample document.", "Another example text."]
    embeddings = generate_embeddings(sample_texts, 'sentence-transformers/all-MiniLM-L6-v2')
    print(embeddings)
