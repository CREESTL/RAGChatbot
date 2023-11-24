## Overview
This repository contains examples of [RAG](https://python.langchain.com/docs/use_cases/question_answering/) systems for answering questions about local .pdf files  

## Functionality
`rag_local_retrieval.py`  
+ Load, embed and store .pdf file in vector DB
+ GPT4All local LLM based on `mistral-7b` model
+ Custom prompt
+ [RetrievalQA](https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#retrievalqa) chain

`rag_api_hf.py`
+ Load, embed and store .pdf file in vector DB
+ Remote HuggingFace LLM based on `zephyr-7b` model. Access via API
+ Custom prompt
+ [RetrievalQA](https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa#retrievalqa) chain

`rag_api_openai_chat.py`
+ Load, embed and store .pdf file in vector DB
+ Remote OpenAI LLM based on `GPT3.5-Turbo` model. Access via paid API
+ Chat memory. Enhanced conversational capabilities compared to chains above
+ Custom prompt
+ [ConversationalRetrievalChain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html?highlight=conversationalretrievalchain#) chain
