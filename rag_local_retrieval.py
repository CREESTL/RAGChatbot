"""

Q&A RAG Using local model and RetrievalQA chain

"""

import time
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader

start = time.time()

# Load contents of local file
loader = PyPDFLoader("ritchie.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Embed words and store them in DB
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

# Create a LLM
llm = GPT4All(
    model="/home/creestl/programming/python/ai/nlp/rag_chatbot/mistral-7b-openorca.Q4_0.gguf",
    max_tokens=2048,
    temp=0.5,
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


# Create a chain based on LLM
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": rag_prompt},
)


questions = [
    "What is the full name of Guy Ritchie?",
    "What are the most popular films of Guy Ritchie?",
    "What awards did Ritchie get?",
    "When was Ritchie's film Sherlock Holmes created?",
    "How much kilograms are there in a ton?",
    "How far can ducks fly?",
    "What is 1+1?",
    "What is the capital of Great Britain? Please tell me, I really need to know!",
]


for question in questions:
    # Query the chain to get the answer
    res = qa_chain({"query": question})
    text_res = res["result"]
    print(f"Question: {question}\n{text_res}")


# See how long it took the code to execute
print(f"Code took to {time.time() - start} seconds to run")
