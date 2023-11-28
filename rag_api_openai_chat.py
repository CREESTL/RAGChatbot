"""

Q&A RAG Using API for OpenAI models and storing chat history

"""

import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings, CacheBackedEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

# TODO or from llms import OpenAI???
from langchain.chat_models import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from dotenv import find_dotenv, load_dotenv


# Load OpenAI API token
load_dotenv(find_dotenv())

start = time.time()

# Load contents of local file
loader = PyPDFLoader("ritchie.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# File to store cached embeddings
cache_file = LocalFileStore("./cache/")
embeddings = GPT4AllEmbeddings()
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=cache_file,
    namespace="some_namespace",
)

# Embed words and store them in DB
vectorstore = Chroma.from_documents(all_splits, cached_embedder)

llm = ChatOpenAI(
    model_name="gpt-e.5-turbo",
    temperature=0.5,  # default is 0.7
    max_tokens=2048,
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
