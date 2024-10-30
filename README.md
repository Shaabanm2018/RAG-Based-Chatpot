# Intelligent-Q-A-Chatbot-Using-Local-RAG-agent-with-LLaMA3

![RAG](https://github.com/user-attachments/assets/8a909e7f-31f6-4691-b579-889ccec6338e)

# Project Description
The Retrieval Augmented Generation (RAG) system leverages three advanced techniques to enhance the accuracy and relevance of generated answers:

1- Routing: The system intelligently routes questions to either a vector store, Elasticsearch, or web search, depending on the content. If the question is related to the vector store, the query is directed there. If it pertains to Elasticsearch, the question is routed accordingly, otherwise, it falls back to web search.

2- Fallback Mechanism: After the initial routing to the vector store, the retrieved documents are graded for relevance to the question. If they are deemed irrelevant, the system falls back to web search. If the query is directed to Elasticsearch, the search results are used without fallback.

3- Sub-correction: Once an answer is generated, the system checks for hallucinations and evaluates if the response is relevant to the original question. If inaccuracies or irrelevance are detected, the system falls back to web search. This phase is bypassed when responses are generated from Elasticsearch.
