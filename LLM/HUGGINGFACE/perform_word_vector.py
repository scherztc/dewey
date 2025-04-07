from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding_model = HuggingFaceEmbedding(model_name= "BAAI/bge-small-en-v1.5")

print(embedding_model._model.device)
