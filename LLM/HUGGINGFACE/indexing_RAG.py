from llama_index.core import VectorStoreIndex


index = VectorStoreIndex.from_documents(
    documents,
    embed_model = embedding_model,
)

index.storge_context.persist(persist_dir="./huggingfaceembeddings")

for doc in index.docstore.docs.values():
    print("Document ID:", doc.ref_doc_id)
    print("Text Chunk:", doc.text)
    print("=" *50)

