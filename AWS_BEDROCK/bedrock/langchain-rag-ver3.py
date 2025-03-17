# Import libraries
import boto3
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain

# Define vectorstore as a global variable
vectorstore_faiss = None  

# Define convenience functions
def config_llm():
    """Configures and returns a Bedrock LLM instance."""
    client = boto3.client('bedrock-runtime')

    model_id = "anthropic.claude-instant-v1"  # Change if necessary
    llm = BedrockLLM(
        model_id=model_id,
        client=client,
        model_kwargs={
            "max_tokens_to_sample": 512,
            "temperature": 0.1,
            "top_p": 1
        }
    )
    return llm

def config_vector_db(filename):
    """Configures FAISS vector database from a PDF file."""
    global vectorstore_faiss  # Ensure global access

    client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(client=client)
    
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)

def vector_search(query):
    """Performs a similarity search in FAISS and returns relevant info."""
    if vectorstore_faiss is None:
        raise ValueError("Vector database is not initialized. Call config_vector_db() first.")

    docs = vectorstore_faiss.similarity_search_with_score(query)
    return "\n".join(doc[0].page_content for doc in docs)

# Configuring the LLM and vector store
llm = config_llm()
config_vector_db("social-media-training.pdf")  

# Define the prompt template
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from an employee. 
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Only answer the question. Do not say things like 
"according to the training or handbook or according to the information provided...".
    
    <Information>
    {info}
    </Information>
    
    {input}

Assistant:
"""

# Configure the prompt template
prompt_template = PromptTemplate(
    input_variables=['input', 'info'],
    template=my_template
)

# Create the LLM chain
question_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    output_key="answer"
)

# Get question, perform similarity search, invoke model, and return result
while True:
    question = input("\nAsk a question about the social media training manual:\n")

    # Perform a similarity search
    info = vector_search(question)

    # Invoke the model, providing additional context
    output = question_chain.invoke({'input': question, 'info': info})

    # Display the result
    print(output['answer'])

