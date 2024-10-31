# Intelligent-Q-A-Chatbot-Using-Local-RAG-agent-with-LLaMA3
An advance question-answering system that combines local document retrieval, Elasticsearch , and web search capabilities using the LLaMA3.2 language model for accurate and contextual responses.

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
The entry point of the system featuring a clean, intuitive interface where users can input their questions.

![Screenshot 2024-10-30 120133](https://github.com/user-attachments/assets/790f3451-79a9-41a7-9248-66f4380b5822)

## Vector Store Routing
When a question is asked, the system first evaluates whether belong to  the locally stored knowledge in the vector database. The process follows these steps:

1. Question Analysis: The input is embedded and compared against stored document vectors
2. Relevance Matching: System identifies the most similar documents in the vector store
3. Response Generation: LLaMA3 processes the retrieved context to generate an answer
4. Quality Check: Response undergoes verification for accuracy and relevance

If the confidence score falls below the threshold, the system automatically triggers the fallback mechanism to web-search.

![Screenshot 2024-10-30 125238](https://github.com/user-attachments/assets/4575c01d-5405-4828-b7ce-a5a68ac81e88)

## ElasticSearch Routing
Integration with Elasticsearch (Network Management System) for specialized queries, particularly focused on network management system data. This routing path is activated when questions pertain to system-specific information.

### Query Device Status from Elasticsearch (Network Management System)
Provides real-time device status information by querying the Elasticsearch database. The system formats complex queries to retrieve current operational states, previous operational status, performance metrics, and health indicators of network devices.

![Screenshot 2024-10-30 132756](https://github.com/user-attachments/assets/8315fd47-b387-4453-b9d5-ef896cd65320)

### Query Device Report from Elasticsearch (Network Management System)
Generates comprehensive device reports by aggregating historical data and performance metrics from Elasticsearch. The system processes this data to create meaningful visualizations and trend analyses.
![Screenshot 2024-10-30 125914](https://github.com/user-attachments/assets/e54b8b4c-a2e9-460a-9f3c-2ffdcf3c04fa)


Upon clicking on the Graph, a wider view will be displayed, offering detailed visual analytics and interactive data exploration capabilities.

![Screenshot 2024-10-30 125930](https://github.com/user-attachments/assets/78e7d3d8-8254-40d3-9a9a-2ebf0000934f)

## Web-Search Routing
The final fallback mechanism that activates when neither the vector store nor Elasticsearch can provide satisfactory answers. This routing:

1. Triggers web search using the Tavily API
2. Filters and ranks search results for relevance
3. Processes retrieved information through LLaMA3.2
4. Generates comprehensive responses based on current web data

This ensures that even questions outside the local knowledge base receive accurate, up-to-date answers.

![Screenshot 2024-10-30 135950](https://github.com/user-attachments/assets/f85238d7-d334-4ccc-bf39-dbe6876b183b)
