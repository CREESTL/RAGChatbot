"""

Q&A RAG Using local model and storing chat history

"""

import time
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import GPT4All
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

# Prompt for LLM to answer the questions
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

conv_qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": rag_prompt},
)

# These are specifically for memory
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
