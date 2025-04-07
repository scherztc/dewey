from llama_index.core import SimpleDirectoryReader

loader = SimpleDirectoryReader(
    input_dir = "/Users/scherztc/Desktop",
    recursive = True,
    required_exts = [".pdf"],
)

documents = loader.load_data()
documents
