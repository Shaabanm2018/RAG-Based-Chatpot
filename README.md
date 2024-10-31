# Intelligent-Q-A-Chatbot-Using-Local-RAG-agent-with-LLaMA3
![RAG](https://github.com/user-attachments/assets/61b1245b-b955-4fbb-99db-7d41cdd7f96b)

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
![Screenshot 2024-10-30 120133](https://github.com/user-attachments/assets/790f3451-79a9-41a7-9248-66f4380b5822)

## Vectore Store Routing
![Screenshot 2024-10-30 125238](https://github.com/user-attachments/assets/4575c01d-5405-4828-b7ce-a5a68ac81e88)

## ElasticSearch Routing
### Query Device Status form Elasticsearch  (Network Management System)
![Screenshot 2024-10-30 132756](https://github.com/user-attachments/assets/8315fd47-b387-4453-b9d5-ef896cd65320)

### Query Device Report form Elasticsearch  (Network Management System)
![Screenshot 2024-10-30 125914](https://github.com/user-attachments/assets/c9a884fb-4901-4bc3-91ce-56e9539ab7c9)

Upon Clicking on the Graph Wider View will be displayed 
![Screenshot 2024-10-30 125930](https://github.com/user-attachments/assets/78e7d3d8-8254-40d3-9a9a-2ebf0000934f)

## Web-Search Routing
![Screenshot 2024-10-30 135950](https://github.com/user-attachments/assets/f85238d7-d334-4ccc-bf39-dbe6876b183b)
