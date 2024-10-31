# Intelligent-Q-A-Chatbot-Using-Local-RAG-agent-with-LLaMA3
A sophisticated question-answering system that combines local document retrieval, Elasticsearch, and web search capabilities using the LLaMA3 language model for accurate and contextual responses.

![RAG](https://github.com/user-attachments/assets/61b1245b-b955-4fbb-99db-7d41cdd7f96b)

## Initial Screen
The entry point of the system featuring a clean, intuitive interface where users can input their questions. The interface displays the system's current status, active routing mechanisms, and provides immediate feedback on query processing.

![Screenshot 2024-10-30 120133](https://github.com/user-attachments/assets/790f3451-79a9-41a7-9248-66f4380b5822)

## Vector Store Routing
When a question is asked, the system first evaluates whether it can be answered using locally stored knowledge in the vector database. The process follows these steps:

1. Question Analysis: The input is embedded and compared against stored document vectors
2. Relevance Matching: System identifies the most similar documents in the vector store
3. Response Generation: LLaMA3 processes the retrieved context to generate an answer
4. Quality Check: Response undergoes verification for accuracy and relevance

If the confidence score falls below the threshold, the system automatically triggers the fallback mechanism.

![Screenshot 2024-10-30 125238](https://github.com/user-attachments/assets/4575c01d-5405-4828-b7ce-a5a68ac81e88)

## ElasticSearch Routing
Integration with Elasticsearch for specialized queries, particularly focused on network management system data. This routing path is activated when questions pertain to system-specific information.

### Query Device Status from Elasticsearch (Network Management System)
Provides real-time device status information by querying the Elasticsearch database. The system formats complex queries to retrieve current operational states, performance metrics, and health indicators of network devices.

![Screenshot 2024-10-30 132756](https://github.com/user-attachments/assets/8315fd47-b387-4453-b9d5-ef896cd65320)

### Query Device Report from Elasticsearch (Network Management System)
Generates comprehensive device reports by aggregating historical data and performance metrics from Elasticsearch. The system processes this data to create meaningful visualizations and trend analyses.

![Screenshot 2024-10-30 125914](https://github.com/user-attachments/assets/c9a884fb-4901-4bc3-91ce-56e9539ab7c9)

Upon clicking on the Graph, a wider view will be displayed, offering detailed visual analytics and interactive data exploration capabilities.

![Screenshot 2024-10-30 125930](https://github.com/user-attachments/assets/78e7d3d8-8254-40d3-9a9a-2ebf0000934f)

## Web-Search Routing
The final fallback mechanism that activates when neither the vector store nor Elasticsearch can provide satisfactory answers. This routing:

1. Triggers web search using the Tavily API
2. Filters and ranks search results for relevance
3. Processes retrieved information through LLaMA3
4. Generates comprehensive responses based on current web data

This ensures that even questions outside the local knowledge base receive accurate, up-to-date answers.

![Screenshot 2024-10-30 135950](https://github.com/user-attachments/assets/f85238d7-d334-4ccc-bf39-dbe6876b183b)
