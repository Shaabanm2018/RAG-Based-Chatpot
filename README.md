# Intelligent-Q-A-Chatbot-Using-Local-RAG-agent-with-LLaMA3

![RAG](https://github.com/user-attachments/assets/8a909e7f-31f6-4691-b579-889ccec6338e)

# Project Description
The Retrieval Augmented Generation (RAG) system leverages three advanced techniques to enhance the accuracy and relevance of generated answers:

1- Routing: The system intelligently routes questions to either a vector store, Elasticsearch, or web search, depending on the content. If the question is related to the vector store, the query is directed there. If it pertains to Elasticsearch, the question is routed accordingly, otherwise, it falls back to web search.

2- Fallback Mechanism: After the initial routing to the vector store, the retrieved documents are graded for relevance to the question. If they are deemed irrelevant, the system falls back to web search. If the query is directed to Elasticsearch, the search results are used without fallback.

3- Sub-correction: Once an answer is generated, the system checks for hallucinations and evaluates if the response is relevant to the original question. If inaccuracies or irrelevance are detected, the system falls back to web search. This phase is bypassed when responses are generated from Elasticsearch.

# Setup 
- pip install langchain-nomic
- ollama pull llama3.2:3b-instruct-fp16 
- pip install -U langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph tavily-python

# Example Usage

## Initial Screen
![Screenshot 2024-10-30 120133](https://github.com/user-attachments/assets/aeb2fa0f-664c-40c3-b0ba-5892ced2fffc)

## Vectore Store Routing
![Screenshot 2024-10-30 125238](https://github.com/user-attachments/assets/51c38ced-e3e7-4982-adfd-36635f45ff50)

## ElasticSearch Routing
### Query Device Status form Elasticsearch  (Network Management System)
![Screenshot 2024-10-30 132756](https://github.com/user-attachments/assets/88053c5b-d963-4195-abe7-9d31e2542e6f)

### Query Device Report form Elasticsearch  (Network Management System)
![Screenshot 2024-10-30 125914](https://github.com/user-attachments/assets/c9a884fb-4901-4bc3-91ce-56e9539ab7c9)

Upon Clicking on the Graph Wider View will be displayed 
![Screenshot 2024-10-30 125930](https://github.com/user-attachments/assets/623a9918-2af0-4ee4-a395-995d6ffa587d)

## Web-Search Routing
![Screenshot 2024-10-30 135950](https://github.com/user-attachments/assets/f9ee5ed3-c316-433c-aff6-1c962423772c)
