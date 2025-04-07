from transformers import AutoModelForCasualLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
import torch

device = (
   torch.device("cuda") if torch.cuda.is_available() else
   torch.device("mps") if torch.mps.is_available() else
   torch.device("cpu")
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCasualLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)

huggingface_llm = HuggingFaceLLM (
      model = model,
      tokenizer = tokenizer,
)

query_engine = index.as_query_engine(llm = huggingface_llm)
