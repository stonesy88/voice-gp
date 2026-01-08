from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-mpnet-base-v2')

# Generate Vector
vector = model.encode("sharp chest pain").tolist()

# Print it formatted for Cypher
print(str(vector))