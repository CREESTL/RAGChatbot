"""

Q&A RAG Using API for Hugging Face models and storing chat history

"""

import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from dotenv import find_dotenv, load_dotenv
import os


load_dotenv(find_dotenv())
HF_TOKEN = os.environ["HF_TOKEN"]

start = time.time()

# Load contents of local file
loader = PyPDFLoader("ritchie.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Embed words and store them in DB
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

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
    """
    Do not use prior knowledge. Answer my question at the end based only on the context provided below:
    
    Context: 
    {context}
    
    In the answer use the same language as in the question.
    If you do not know the answer, say "I do not know the answer". Do not make the answer up.
    Your answer must consist of maximum of 3 sentences. Start your answer with a new line and put [ANSWER] before it.

    Question: {question}
    
    Your answer:
    """
)

# Create memory for chat
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Create a chain based on LLM
conv_qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": rag_prompt},
)


# WARNING: Using to many requests in a short period of type may lead to Server Error of HF
questions = [
    "When was Ritchie born?",
    "What are his most popular films?",
    "What awards did Ritchie get?",
    "What is the first one?",
    "When was his film Sherlock Holmes created?",
    "Who was the main actor?",
    "How much kilograms are there in a ton?",
]

for question in questions:
    # Query the chain to get the answer
    res = conv_qa_chain({"question": question})
    text_res = res["answer"]
    print(f"Question: {question}\n{text_res}")


# See how long it took the code to execute
print(f"Code took to {time.time() - start} seconds to run")
