"""

Q&A RAG Using local model

"""

import time
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import MathpixPDFLoader

start = time.time()

# Load contents of local file
loader = PyPDFLoader("ritchie.pdf")
pages = loader.load()

# Embed words and store them in DB
vectorstore = Chroma.from_documents(documents=pages, embedding=GPT4AllEmbeddings())

# Create a LLM
llm = GPT4All(
    model="/home/creestl/programming/python/ai/nlp/rag_chatbot/mistral-7b-openorca.Q4_0.gguf",
    max_tokens=4056,
    temp=0.5,
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
    print(f"Question: {question}\n{text_res}")


# See how long it took the code to execute
print(f"Code took to {time.time() - start} seconds to run")
