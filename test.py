from pyllamacpp import load_model, generate_embeddings, tokenize, generate_from_tokens
import numpy as np

def cosine_similarity(a, b):
    """
    Compute the cosine similarity between two vectors a and b.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim

load_model("models/7B/ggml-model-q4_0.bin")

TXT = "I would like to know more about engineering. Also biology"
tokens = tokenize(TXT)
embeddings = generate_from_tokens(tokens)


for s in ["Computer", "Science", "History", "Literature"]:
    print(f'{s}: {cosine_similarity(embeddings, generate_embeddings(s))}')

# print("Generating embeddings...")

# base_str = "Processor"
# base = generate_embeddings(base_str)

# similarities = {}
# for test_str in ["Computer", "Medical equipement", "Literature", "History"]:
#     print(f'Generating embedding for {test_str}')
#     embedding = generate_embeddings(test_str)
#     similarities[test_str] = cosine_similarity(base, embedding)

# print(f"Similarities with {base_str}")
# for key, value in similarities.items():
#     print(f'{key}: {value}')

