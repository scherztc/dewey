while True:
   question = input("Questions:")
   if question.lower() == "quit": break
   print(query_engine.query(question).response)


from langchain_openai import ChatOpenAI

load_dotenv()

os.environ["OPEN_API_KEY"] = os.getenv("OpenAI_token")
openi_llm = ChatOpenAI(temperature = 0.7, model_name = "gpt-40-mini")

query_engine = index.as_query_engine(llm = openai_llm)



