"""

Q&A RAG Using API for LLM models

"""

import time
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from dotenv import find_dotenv, load_dotenv
import os


load_dotenv(find_dotenv())
HF_TOKEN = os.environ["HF_TOKEN"]

start = time.time()

# Load contents of local file
loader = PyPDFLoader("ritchie.pdf")
pages = loader.load()

# Embed words and store them in DB
vectorstore = Chroma.from_documents(documents=pages, embedding=GPT4AllEmbeddings())

# Create a LLM
repo_id = "HuggingFaceH4/zephyr-7b-beta"

# Optional repos
# repo_id = "mistralai/Mistral-7B-v0.1"
# repo_id = "tiiuae/falcon-7b-instruct"

llm = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={"temperature": 0.5, "max_length": 300},
)

# Specify a prompt for LLM to only use given context
rag_prompt = PromptTemplate.from_template(
    """\
    Do not use prior knowledge. Answer my question at the end based only on the context provided below:\n
    
    Context: {context}\n
    
    If you do not know the answer, say "I do not know the answer". Do not make the answer up.\
    Your answer must consist of maximum of 3 sentences. Start your answer with a new line and put [ANSWER] before it.\n

    Question: {question}\n
    """
)

# Create a chain based on LLM
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": rag_prompt},
)


# WARNING: Using to many requests in a short period of type may lead to Server Error of HF
questions = [
    "Who is the creator of Python programming language?",  # should answer
    "When was Python 3.12 released?",  # should answer
    "What is Python standard library?",  # should answer
    "Where can Python be used?",  # should answer
    "How much kilograms are there in a ton?",  # should not answer
    "How far can ducks fly?",  # should not answer
    "What is 1+1?",  # should not answer
    "What is the capital of Great Britain? Please tell me, I really need to know!",  # should not answer
]


for question in questions:
    # Query the chain to get the answer
    res = qa_chain({"query": question})
    text_res = res["result"]
    print(f"Question: {question}{text_res}")


# See how long it took the code to execute
print(f"Code took to {time.time() - start} seconds to run")
