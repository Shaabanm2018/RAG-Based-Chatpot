from flask import Flask, render_template, request, jsonify
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.docstore.document import Document
import json
from langchain_core.messages import HumanMessage, SystemMessage
import os
from pprint import pprint
import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END
from langgraph.graph import StateGraph
from IPython.display import Image, display
from elasticsearch import Elasticsearch
from datetime import datetime
from datetime import datetime, timedelta
import base64
from PIL import Image
from PIL import Image
import io
import re
import yaml
import torch
import pytz
import pandas as pd
import numpy as np
import matplotlib
import shutil
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None 
app = Flask(__name__)

# Set CUDA device to GPU 0 (RTX 4050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0, num_gpu=1)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json", num_gpu=1)

import getpass
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
os.environ["TAVILY_API_KEY"] = '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'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["LANGCHAIN_API_KEY"] = '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'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"

def create_retriever():
    # List of paths to your text files
    file_paths = [
        "rag_documents/Kibana installation and Setup.md",
        "rag_documents/Elasticsearch Installation & Setup Wit.md",
    ]

    # List of URLs
    urls = [
        "https://phoenixnap.com/kb/install-kubernetes-on-ubuntu",
    ]

    # Initialize docs_list
    docs_list = []

    # Process local Markdown files
    for file_path in file_paths:
        if os.path.exists(file_path):

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                docs_list.append(Document(page_content=content, metadata={"source": file_path}))
        else:
            print(f"Warning: File not found - {file_path}")

    # Process URLs
    for url in urls:
        try:
            web_docs = WebBaseLoader(url).load()
            docs_list.extend(web_docs)
        except Exception as e:
            print(f"Error loading URL {url}: {str(e)}")


    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs_list)
    embeddings = NomicEmbeddings(
        model="nomic-embed-text-v1.5",
        inference_mode="local",
        device="cuda:NVIDIA GeForce RTX 4050 Laptop GPU"  # Updated device specification
    )
    # Add to vectorDB
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )

    # Create retriever
    retriever = vectorstore.as_retriever(k=3)
    return retriever
retriever = create_retriever()

### Router
router_instructions = """You are not supposed to answer or generate code in response to the question. Your only task is to determine whether the user question should be routed to a vectorstore, web search, or elasticsearch query based on the following rules:

1. Use 'vectorstore' IF AND ONLY IF the question is specifically about Elasticsearch, Kibana, or Kubernetes documentation and features.

2. Use 'elasticsearch_query' IF the question include one of the following conditions:
1- the question contains the word "status" and an actual ip address
2- the question contains the word "status" and the word "links" 
3- the question contains the word "status" and the word "interface" or "interfaces".

3. Use 'web_search' for ALL other topics, including but not limited to:
   - Current events
   - Technology in general
   - AI and machine learning
   - Agent systems
   - Any topic not specifically about Elasticsearch, Kibana, or Kubernetes.

Do not generate or return anything else besides a JSON object with a single key 'datasource'.""" 

### Reteriver Grader
# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

### Generate
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

generate a complete and concise answer.

Answer:"""


# Post-processing
def format_docs(docs):
    if isinstance(docs, str):
        return docs
    return "\n\n".join(doc.page_content for doc in docs)

### Hallucination Grader

# Hallucination grader instructions
# Updated Hallucination grader instructions
hallucination_grader_instructions = """
You are a teacher grading a quiz with a focus on evaluating potential hallucinations in student answers. 
You will be given FACTS and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the core of the STUDENT ANSWER is grounded in the FACTS provided. 
(2) Allow for reasonable inferences or related knowledge that, while not explicitly stated in the FACTS, are logically connected and don't contradict the given information.
(3) Be cautious of information that seems completely unrelated or contradictory to the FACTS.

Score:
- A score of "yes" means that the student's answer is largely grounded in the FACTS, with any additional information being reasonable and related.
- A score of "no" means that the student's answer contains significant information that is unrelated or contradictory to the FACTS.

Explain your reasoning in a step-by-step manner, considering both the information directly from the FACTS and any reasonable inferences the student might have made.
In your explanation, highlight what parts of the answer are grounded in the FACTS, what parts are reasonable inferences, and what parts (if any) seem to be potential hallucinations.
"""

# Grader prompt
hallucination_grader_prompt = """
FACTS: 

{documents}

STUDENT ANSWER: {generation}

Return JSON with two keys:
1. 'binary_score': 'yes' or 'no' to indicate whether the STUDENT ANSWER is sufficiently grounded in the FACTS.
2. 'explanation': a detailed explanation of your scoring decision, including analysis of factual content, reasonable inferences, and potential hallucinations.
"""
### Answer Grader

# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""


### Network Info Extractor
# Instructions for extracting IP, Date, and Time
network_info_extractor_instructions = """
You are a smart assistant that extracts information from a user question. Your task is to extract the following:
- IP address
- Date (if mentioned)
- Time (if mentioned)

Make sure to return the results in a JSON format with the keys:
- 'ip_address': containing the extracted IP address.
- 'date': containing the extracted date in YYYY-MM-DD format or null if not provided.
- 'time': containing the extracted time in HH:MM:SS (24-hour) format or null if not provided.

Return "null" for any field that is not mentioned. Always follow the formatting conventions strictly.
"""

# Prompt Template for extraction
network_info_extractor_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
You are a smart assistant that extracts information from a user question. Extract the following details from the question:
- IP address (if provided)
- Date (if provided)
- Time (if provided)
- Interface Name (if provided)
- Link Name (if provided)

If any information is missing, return the missing fields with a null value. 
Return the output in the following JSON format with the keys 'ip_address', 'date', and 'time':

{{
    "ip_address": "extracted IP",
    "date": "extracted date in YYYY-MM-DD format or null",
    "time": "extracted time in HH:MM:SS format or null",
    "interface_name": "extracted interface name",
    "link_name": "extracted link name"
}}

Important: Always format the date as YYYY-MM-DD and the time as HH:MM:SS (24-hour format).
For example, "July 7, 2023" should be output as "2023-07-07", and "5pm" as "17:00:00".

Here is the user question: {question} 
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt_network = """
You have the following status information:

Context: {context}

Question: {question}

Assume that the date and time mentioned in the question are the same as the status information provided in the context, even if the context does not specify any date and time. If no specific date or time is mentioned in the question, assume it refers to now.

Please answer the question without any explanation or justification. Provide the answer in the following format:

"The status of the [device] with IP address [IP address] [time reference] is/was [status]." 
If the context includes uncertainty, respond with both possible statuses.
"""
def router_logic(question):
    """
    Test function to verify router logic based on given rules
    Returns: Expected routing destination and explanation
    """
    # Convert question to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # Rule 1: Vectorstore check - now more specific to installation and setup
    installation_keywords = ['install', 'setup', 'configuration', 'configure', 'deployment']
    products = ['elasticsearch', 'kibana', 'kubernetes']
    
    # Check if question is about installation/setup of specific products
    is_installation_query = (
        any(install_kw in question_lower for install_kw in installation_keywords) and
        any(product in question_lower for product in products)
    )
    
    # Rule 2: Elasticsearch query checks
    has_status = 'status' in question_lower
    has_report = 'report' in question_lower
    has_interface = 'interface' in question_lower or 'interfaces' in question_lower
    has_links = 'links' in question_lower
    has_ip = bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', question))
    
    # Determine routing
    if is_installation_query:
        matched_product = next((prod for prod in products if prod in question_lower), '')
        matched_action = next((kw for kw in installation_keywords if kw in question_lower), '')
        return {
            'datasource': 'vectorstore'
        }
    elif has_status and (has_ip or has_links or has_interface) or has_report:
        return {
            'datasource': 'elasticsearch_query'
        }
    else:
        return {
            'datasource': 'web_search'
        }
    
### Search
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

# Preprocessing function (unchanged)
def preprocess_question(question):
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(1)).strftime("%Y-%m-%d")
    question = question.replace("today", today)
    question = question.replace("yesterday", yesterday)
    
    def replace_date(match):
        day = int(match.group(1))
        month = match.group(2)
        year = match.group(3) if match.group(3) else str(datetime.now().year)
        try:
            date_obj = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            return match.group(0)

    question = re.sub(r"\b(\d{1,2}) (\w+)(?: (\d{4}))?\b", replace_date, question)
    
    def replace_time(match):
        time_str = match.group(1)
        period = match.group(2).lower()
        try:
            time_obj = datetime.strptime(f"{time_str} {period}", "%I%p")
            return time_obj.strftime("%H:%M:%S")
        except ValueError:
            return match.group(0)

    question = re.sub(r"\b(\d{1,2})(am|pm)\b", replace_time, question)
    
    return question

# Updated post-processing function
def postprocess_result(result):
    current_datetime = datetime.now()

    # Handle date
    if result['date'] is None or result['date'] == 'null':
        result['date'] = current_datetime.strftime("%Y-%m-%d")
    elif '-' not in result['date']:
        try:
            date_obj = datetime.strptime(result['date'], "%m/%d/%Y")
            result['date'] = date_obj.strftime("%Y-%m-%d")
        except ValueError:
            result['date'] = current_datetime.strftime("%Y-%m-%d")

    # Handle time
    if result['time'] is None or result['time'] == 'null':
        result['time'] = current_datetime.strftime("%H:%M:%S")
    elif ':' not in result['time']:
        try:
            time_obj = datetime.strptime(result['time'], "%I%p")
            result['time'] = time_obj.strftime("%H:%M:%S")
        except ValueError:
            result['time'] = current_datetime.strftime("%H:%M:%S")

    return result

def get_ifalias(date=None, time=None, if_alias=None):
    """
    Query device status from Elasticsearch based on IP address and/or interface alias.
    """
    # Handle legacy argument order (ip_address, date, time)
    if date and time and isinstance(date, str) and isinstance(time, str):
        datetime_str = f"{date}T{time}"
    elif isinstance(ip_address, str) and isinstance(date, str) and isinstance(time, str):
        datetime_str = f"{date}T{time}"
    else:
        raise ValueError("Invalid argument combination. Required format: (ip_address, date, time) or (date=date, time=time, ip_address=ip_address)")

    # Input validation
    if not if_alias:
        return "At least one of if_alias must be provided"

    try:
        # Convert the datetime string into a datetime object
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
        # Calculate the time one hour earlier
        one_day_earlier = datetime_obj - timedelta(hours=24)
    except ValueError as e:
        raise ValueError(f"Invalid date/time format. Expected YYYY-MM-DD and HH:MM:SS, got date={date}, time={time}") from e

    # Certificate path - using Windows path format
    cert_path = "certs\ca.crt"
    
    # Verify certificate file exists
    if not os.path.exists(cert_path):
        return f"Certificate file not found at: {cert_path}"

    try:
        # Connect to Elasticsearch using SSL configuration
        es = Elasticsearch(
            ['https://172.23.37.31:30920'],
            basic_auth=('elastic', '1Qgd0A1BkFxmVuqZ'),
            ssl_assert_hostname=False,
            ssl_assert_fingerprint=None,
            ca_certs=cert_path
        )
    except Exception as e:
        return f"Failed to connect to Elasticsearch: {str(e)}"

    # Rest of the query logic remains the same...
    must_conditions = [
        {
            "range": {
                "@timestamp": {
                    "gte": one_day_earlier.isoformat(),
                    "lte": datetime_obj.isoformat(),
                    "format": "strict_date_hour_minute_second",
                    "time_zone": "+08:00"
                }
            }
        }
    ]
    
    if if_alias:
        if '*' in if_alias:
            must_conditions.append({"wildcard": {"ifAlias": if_alias.lower()}})
        else:
            must_conditions.append({"match": {"ifAlias": if_alias}})

    query = {
        "_source": ["@timestamp", "ifAlias"],
        "query": {
            "bool": {
                "must": must_conditions
            }
        },
        "sort": [
            {
                "@timestamp": {
                    "order": "desc"
                }
            }
        ],
        "script_fields": {
            "local_timestamp": {
                "script": {
                    "lang": "painless",
                    "source": "doc['@timestamp'].value.plusHours(8)"
                }
            }
        },
        "size": 100
    }

    try:
        response = es.search(index="snmp-data-*", body=query)
    except Exception as e:
        return f"Error executing search: {str(e)}"

    # Process results
    formatted_timestamp = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
    latest_status_per_ip = {}
    unique_if_aliases = set()
    for hit in response['hits']['hits']:
        if '_source' in hit:
            current_if_alias = hit['_source'].get('ifAlias', 'Unknown Interface')
            unique_if_aliases.add(current_if_alias)
    return list(unique_if_aliases)

def get_interface_traffic(if_alias, date, time):
    """
    Query interface traffic data from Elasticsearch for a specific interface over 24 hours.
    
    Args:
        if_alias (str): Interface alias name
        date (str): Date in YYYY-MM-DD format
        time (str): Time in HH:MM:SS format
    
    Returns:
        list: List of dictionaries containing timestamp and traffic data
    """
    try:
        # Convert input date and time to datetime object
        datetime_str = f"{date}T{time}"
        end_time = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
        # Convert to UTC for Elasticsearch
        kl_tz = pytz.timezone('Asia/Kuala_Lumpur')
        end_time = kl_tz.localize(end_time).astimezone(pytz.UTC)
        # Calculate start time (24 hours earlier)
        start_time = end_time - timedelta(hours=24)
    except ValueError as e:
        raise ValueError(f"Invalid date/time format. Expected YYYY-MM-DD and HH:MM:SS, got date={date}, time={time}") from e
    cert_path = "certs\ca.crt"
    
    # Verify certificate file exists
    if not os.path.exists(cert_path):
        return f"Certificate file not found at: {cert_path}"
    # Connect to Elasticsearch
    es = Elasticsearch(
            ['https://172.23.37.31:30920'],
            basic_auth=('elastic', '1Qgd0A1BkFxmVuqZ'),
            ssl_assert_hostname=False,
            ssl_assert_fingerprint=None,
            ca_certs=cert_path
        )

    # Build the query
    query = {
        "aggs": {
            "time_buckets": {
                "date_histogram": {
                    "field": "@timestamp",
                    "fixed_interval": "30m",
                    "time_zone": "Asia/Kuala_Lumpur",
                    "extended_bounds": {
                        "min": int(start_time.timestamp() * 1000),
                        "max": int(end_time.timestamp() * 1000)
                    }
                },
                "aggs": {
                    "interface_stats": {
                        "multi_terms": {
                            "terms": [
                                {
                                    "field": "host.ip.keyword"
                                },
                                {
                                    "field": "ifAlias.keyword"
                                }
                            ],
                            "size": 100
                        },
                        "aggs": {
                            "in_traffic": {
                                "sum": {
                                    "field": "ifHCInOctets"
                                }
                            },
                            "out_traffic": {
                                "sum": {
                                    "field": "ifHCOutOctets"
                                }
                            }
                        }
                    }
                }
            }
        },
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "ifAlias.keyword": if_alias
                        }
                    }
                ],
                "filter": [
                    {
                        "range": {
                            "@timestamp": {
                                "format": "strict_date_optional_time",
                                "gte": start_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                "lte": end_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                            }
                        }
                    }
                ]
            }
        }
    }

    # Execute the query
    response = es.search(index="snmp-data-*", body=query)

    # Process the results
    results = []
    for bucket in response['aggregations']['time_buckets']['buckets']:
        timestamp = datetime.fromtimestamp(bucket['key'] / 1000, kl_tz)
        
        # Get interface stats
        for interface_bucket in bucket['interface_stats']['buckets']:
            if interface_bucket['key'][1] == if_alias:  # Match the specific interface
                in_traffic = int(interface_bucket['in_traffic']['value'])  # Ensure integer values
                out_traffic = int(interface_bucket['out_traffic']['value'])  # Ensure integer values

                results.append({
                    'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'ip_address': interface_bucket['key'][0],
                    'interface': interface_bucket['key'][1],
                    'in_traffic': in_traffic,
                    'out_traffic': out_traffic
                })

    if not results:
        return f"No data found for interface {if_alias} in the specified time range"
    df = pd.DataFrame(results)
    return df
def calculate_throughput(octets, time_in_seconds):
    # Conversion factors
    tb_factor = 1_099_511_627_776  # 1 TB in octets
    gb_factor = 1_073_741_824       # 1 GB in octets
    mb_factor = 1_048_576           # 1 MB in octets
    kb_factor = 1_024               # 1 KB in octets

    # Convert total octets to TB and calculate throughput in TB/s
    total_tb = octets / tb_factor
    throughput_tb_per_second = total_tb / time_in_seconds

    # Determine the appropriate unit based on the throughput
    if throughput_tb_per_second >= 1.0:
        return f"{round(throughput_tb_per_second, 1)} TB/s"
    else:
        # Check for GB
        throughput_gb_per_second = (octets / gb_factor) / time_in_seconds
        if throughput_gb_per_second >= 1.0:
            return f"{round(throughput_gb_per_second, 1)} GB/s"
        
        # Check for MB
        throughput_mb_per_second = (octets / mb_factor) / time_in_seconds
        if throughput_mb_per_second >= 1.0:
            return f"{round(throughput_mb_per_second, 1)} MB/s"
        
        # Check for KB
        throughput_kb_per_second = (octets / kb_factor) / time_in_seconds
        if throughput_kb_per_second >= 1.0:
            return f"{round(throughput_kb_per_second, 1)} KB/s"
        
        # If none of the above, return bytes
        return f"{octets} B/s"

def convert_to_bytes(value, unit):
    """Convert traffic values to bytes based on unit"""
    unit = unit.upper()
    multipliers = {
        'B/S': 1,
        'KB/S': 1024,
        'MB/S': 1024**2,
        'GB/S': 1024**3,
        'TB/S': 1024**4
    }
    return value * multipliers[unit]

def format_traffic(bytes_value):
    """Format bytes into the most appropriate unit"""
    units = ['B/s', 'KB/s', 'MB/s', 'GB/s', 'TB/s']
    unit_index = 0
    
    while bytes_value >= 1024 and unit_index < len(units) - 1:
        bytes_value /= 1024
        unit_index += 1
    
    return bytes_value, units[unit_index]

def plot_interface_traffic(df, ifalises, traffic_name):
    # Create the 'graphs' directory if it doesn't exist
    
    for i in ifalises:
        # Create a copy of the filtered data
        df_interface = df[df['interface'] == i].copy()
        
        if df_interface.empty:
            print(f"No data available for interface: {i}")
            continue
            
        # Process the data
        df_interface.loc[:, 'timestamp'] = pd.to_datetime(df_interface['timestamp'])
        
        # Extract values using temporary series to avoid warnings
        value_series = df_interface[traffic_name].str.extract(r'([\d.]+)').astype(float)
        unit_series = df_interface[traffic_name].str.extract(r'([A-Za-z/]+)')
        
        df_interface.loc[:, 'traffic_value'] = value_series
        df_interface.loc[:, 'traffic_unit'] = unit_series.iloc[:, 0]

        # Convert all values to bytes
        bytes_values = [convert_to_bytes(val, unit) 
                       for val, unit in zip(df_interface['traffic_value'], 
                                          df_interface['traffic_unit'])]
        
        # Determine the best unit for display
        max_bytes = max(bytes_values)
        _, display_unit = format_traffic(max_bytes)
        
        # Convert bytes back to the display unit
        if display_unit == 'B/s':
            display_values = bytes_values
        else:
            divisor = convert_to_bytes(1, display_unit)
            display_values = [b / divisor for b in bytes_values]

        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Plot with lines only
        plt.plot(df_interface['timestamp'], display_values, color='teal')

        # Add vertical line at midpoint
        middle_date = df_interface['timestamp'].iloc[len(df_interface) // 2]
        plt.axvline(x=middle_date, color='gray', linestyle='--')

        # Customize the graph
        plt.title(f'Sum of HrIfInOctets per Second for Interface: {i}', fontsize=14)
        plt.xlabel('@timestamp per 30 minutes', fontsize=12)
        plt.ylabel(f'Traffic ({display_unit})', fontsize=12)

        # Set dynamic y-axis ticks
        y_min, y_max = min(display_values), max(display_values)
        y_ticks = np.linspace(y_min, y_max, num=5)
        plt.yticks(y_ticks, [f'{tick:.2f}' for tick in y_ticks])

        # Set dynamic x-axis ticks
        x_ticks = pd.date_range(start=df_interface['timestamp'].min(), 
                               end=df_interface['timestamp'].max(), 
                               periods=5)
        plt.xticks(x_ticks, x_ticks.strftime('%H:%M\n%b %d, %Y'))

        # Show major grid lines only
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='lightgray')
        plt.grid(False, which='minor')

        plt.tight_layout()
        plt.savefig(f'graphs/{traffic_name}_{i}.png')  # Save the figure with formatted path
        plt.close()

def display_graph():
    folder_path = r"C:\Users\User\Documents\llm_rag_gui\graphs"
        
    # Get all matching pairs of traffic graphs
    graph_pairs = get_matching_traffic_pairs(folder_path)
    
    if not graph_pairs:
        return "No matching traffic graph pairs found in the specified folder."

    
    
    # Generate the markdown for all pairs
    markdown_images = []
    
    for in_path, out_path in graph_pairs:
        # Create side-by-side image
        combined_img = create_side_by_side_image(in_path, out_path)
        
        if combined_img is None:
            continue
            
        # Convert to base64
        img_buffer = io.BytesIO()
        combined_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Extract interface name for the caption
        interface_name = os.path.basename(in_path)[11:]  # Remove 'in_traffic_' prefix
        
        # Add to markdown list
        markdown_images.append(f"### Traffic for {interface_name}\n![{interface_name}](data:image/png;base64,{img_str})\n")
    
    if not markdown_images:
        return "Failed to process the traffic graph pairs."
    
    # Combine all images into one response
    answer = "Here are your traffic graphs:\n\n" + "\n".join(markdown_images)
    return answer

def get_if_device_status(ip_address=None, date=None, time=None, if_alias=None):
    """
    Query device status from Elasticsearch based on IP address and/or interface alias.

    Args:
        ip_address (str, optional): Host IP address
        date (str): Date in YYYY-MM-DD format
        time (str): Time in HH:MM:SS format
        if_alias (str, optional): Interface alias (supports wildcards *)

    Returns:
        str: Formatted string containing device status information
    """
    # Handle legacy argument order (ip_address, date, time)
    if date and time and isinstance(date, str) and isinstance(time, str):
        datetime_str = f"{date}T{time}"
    elif isinstance(ip_address, str) and isinstance(date, str) and isinstance(time, str):
        datetime_str = f"{date}T{time}"
    else:
        raise ValueError("Invalid argument combination. Required format: (ip_address, date, time) or (date=date, time=time, ip_address=ip_address)")

    # Input validation
    if not ip_address and not if_alias:
        return "At least one of ip_address or if_alias must be provided"

    try:
        # Convert the datetime string into a datetime object
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
        # Calculate the time one hour earlier
        one_hour_earlier = datetime_obj - timedelta(hours=1)
    except ValueError as e:
        raise ValueError(f"Invalid date/time format. Expected YYYY-MM-DD and HH:MM:SS, got date={date}, time={time}") from e
    cert_path = "certs\ca.crt"
    # Connect to Elasticsearch using the hostname
    es = Elasticsearch(
        ['https://172.23.37.31:30920'],
            basic_auth=('elastic', '1Qgd0A1BkFxmVuqZ'),
            ssl_assert_hostname=False,
            ssl_assert_fingerprint=None,
            ca_certs=cert_path
    )

    # Build the must conditions based on provided parameters
    must_conditions = [
        {
            "range": {
                "@timestamp": {
                    "gte": one_hour_earlier.isoformat(),
                    "lte": datetime_obj.isoformat(),
                    "format": "strict_date_hour_minute_second",
                    "time_zone": "+08:00"
                }
            }
        }
    ]

    if ip_address:
        if '*' in ip_address:
            must_conditions.append({"wildcard": {"host.ip": ip_address}})
        else:
            must_conditions.append({"match": {"host.ip": ip_address}})
    
    if if_alias:
        if '*' in if_alias:
            must_conditions.append({"wildcard": {"ifAlias": if_alias.lower()}})  # Convert to lowercase for case-insensitive search
        else:
            must_conditions.append({"match": {"ifAlias": if_alias}})

    # Define the search query
    query = {
        "_source": ["ifDeviceStatus", "@timestamp", "ifAlias", "host.ip"],
        "query": {
            "bool": {
                "must": must_conditions
            }
        },
        "sort": [
            {
                "@timestamp": {
                    "order": "desc"
                }
            }
        ],
        "script_fields": {
            "local_timestamp": {
                "script": {
                    "lang": "painless",
                    "source": "doc['@timestamp'].value.plusHours(8)"
                }
            }
        },
        "size": 100  # Increased from default 10 to match Kibana query
    }

    # Execute the search query
    response = es.search(index="snmp-data-*", body=query)

    # Initialize formatted_timestamp with the provided date and time
    formatted_timestamp = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

    # Initialize a dictionary to store the latest result per IP
    latest_status_per_ip = {}

    # Loop through all hits to collect the latest status per IP
    for hit in response['hits']['hits']:
        if '_source' in hit:
            current_if_alias = hit['_source'].get('ifAlias', 'Unknown Interface')
            current_ip = hit['_source']['host'].get('ip', 'Unknown IP')
            if_device_status = hit['_source'].get('ifDeviceStatus', '')

            # If status is empty, set it to default message
            if not if_device_status:
                if_device_status = "Up or not being configured by the administrator"

            # Extract local timestamp, if available
            if 'fields' in hit and 'local_timestamp' in hit['fields']:
                local_timestamp = hit['fields']['local_timestamp'][0]
                local_timestamp = datetime.strptime(local_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                formatted_timestamp = local_timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # If this IP is not in the dictionary or the current timestamp is more recent, update it
            if current_ip not in latest_status_per_ip or local_timestamp > latest_status_per_ip[current_ip]['timestamp']:
                latest_status_per_ip[current_ip] = {
                    'if_alias': current_if_alias,
                    'status': if_device_status,
                    'timestamp': local_timestamp,
                    'formatted_timestamp': formatted_timestamp
                }

    # If no results found, return appropriate default message
    if not latest_status_per_ip:
        search_criteria = []
        if ip_address:
            search_criteria.append(f"IP: {ip_address}")
        if if_alias:
            search_criteria.append(f"Interface: {if_alias}")
        search_str = ", ".join(search_criteria)
        return f"No data found on the last one hour for the mentioned query"

    # Format the results
    status_results = [
    f"IP: {ip}, Interface: {data['if_alias']}, Status: {data['status']}, Time: {data['formatted_timestamp']}"
    for ip, data in latest_status_per_ip.items()
    ]

    # Initialize an empty string to store the formatted document
    document = ""

    # Loop through each item in the list and process it
    for item in status_results:
        # Initialize a dictionary to store details
        details = {}
        
        # Split the item by commas and process each part
        for part in item.split(', '):
            # Ensure that each part can be split into exactly two elements
            if ': ' in part:
                key, value = part.split(': ', 1)
                details[key.strip()] = value.strip()
            else:
                print(f"Warning: '{part}' does not contain a ': ' delimiter and will be skipped.")
        
        # Check if all required keys are present
        if 'IP' in details and 'Interface' in details and 'Status' in details and 'Time' in details:
            # Extract date and time separately from 'Time' value
            date, time = details['Time'].split(' ')
            formatted_string = (
            f"On {date} at {time}, the device with IP address {details['IP']} "
            f"on the {details['Interface']} interface is currently {details['Status']}.\n"
        )
            # Add the formatted string to the document
            document += formatted_string
        else:
            print("Error: Missing necessary information in the input data.")

    # Print the final document
    print(document)

    # Return the latest statuses as a combined string
    return document



class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents


### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    global route
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    # RAG generation
    # if state['route'] == 'esquery_es_':
    #     documents = "The answer for the question of "+ question + ' is: \n'+ documents
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            
            continue
    return {"documents": filtered_docs, "web_search": web_search}
route = None
def query_elasticsearch(state):
    """
    Query Elasticsearch for documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---QUERY ES---")
    question = state["question"]
    question_preprocess = preprocess_question(question)
    network_info_extractor_prompt_formatted = network_info_extractor_prompt.format(question=question_preprocess)
    # Run the LLM to extract information
    result = llm_json_mode.invoke(
        [SystemMessage(content=network_info_extractor_instructions)]
        + [HumanMessage(content=network_info_extractor_prompt_formatted)]
    )

    # Parse the output JSON
    result = postprocess_result(json.loads(result.content))
    interface_name = '*' + (result.get('interface_name') or result.get('link_name') or '') + '*' if (result.get('interface_name') or result.get('link_name')) else None
    if 'report' in question.lower():
        ifalises = get_ifalias( result['date'], result['time'],interface_name[1:-1])
        dfs = []
        # Loop over each ifAlias and get the DataFrame
        for i in ifalises:
            df = get_interface_traffic(i, result['date'], result['time'])
            dfs.append(df)  # Append each DataFrame to the list
        df_combined = pd.concat(dfs, ignore_index=True)                                                                                                                                                                                                                                           
        df_combined['in_traffic'] = df_combined.apply(lambda row: calculate_throughput(row['in_traffic'],1800), axis=1)
        df_combined['out_traffic'] = df_combined.apply(lambda row: calculate_throughput(row['out_traffic'],1800), axis=1)
        df_combined.to_csv('data.csv',index=False)
        shutil.rmtree('graphs', ignore_errors=True)  # Remove if exists, ignore if doesn't
        os.makedirs('graphs', exist_ok=True) 
        plot_interface_traffic(df_combined, ifalises,"in_traffic")
        plot_interface_traffic(df_combined, ifalises,"out_traffic")
        document= display_graph()

    else:
        document = get_if_device_status(result['ip_address'], result['date'], result['time'], interface_name)
    return {"documents": document, "question": question}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    print(documents)
    return {"documents": documents}


### Edges


def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    global route
    print("---ROUTE QUESTION---")
    source = router_logic(state["question"])['datasource']
    if source == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        route = "websearch"
        return "websearch"
    elif source == 'elasticsearch_query':
        print("---ROUTE QUESTION TO ELASTICSEARCH---")
        route = "query_es"
        return "query_es"
    elif source == "vectorstore":
        print("---ROUTE  QUESTION TO RAG---")
        route = "vectorstore"
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    # hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
    #     documents=format_docs(documents), generation=generation.content
    # )
    # result = llm_json_mode.invoke(
    #     [SystemMessage(content=hallucination_grader_instructions)]
    #     + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    # )
    # grade = json.loads(result.content)["binary_score"]
    grade = "yes"
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"
    

from langgraph.graph import END, StateGraph
workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("query_elasticsearch", query_elasticsearch)  # Node for Elasticsearch query


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
        "query_es": "query_elasticsearch",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
# Path for Elasticsearch query
#workflow.add_edge("query_elasticsearch", "generate")

# Conditional edges after generate
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Compile

#display(Image(graph.get_graph().draw_mermaid_png()))
import re

def reformat_answer(answer):
    # Remove triple backticks and asterisks
    answer = answer.replace("```", "").replace("*", "")
    
    # Define a regex pattern to match different step formats
    step_pattern = re.compile(r'(\d+[\.\-]\s|Step\s\d+:\s|Method\s\d+:\s)', re.IGNORECASE)
    
    # Check if the answer contains steps
    if step_pattern.search(answer):
        # Find the introductory sentence
        intro_match = re.match(r'^(.*?)(\d+[\.\-]\s|Step\s\d+:\s|Method\s\d+:\s)', answer, re.DOTALL)
        if intro_match:
            intro = intro_match.group(1).strip()
            steps = step_pattern.split(answer[intro_match.end()-len(intro_match.group(2)):])  # Adjust to include the first step
        else:
            intro = ""
            steps = step_pattern.split(answer)
        
        # Reconstruct the answer with the introductory sentence and each step on a new line
        reformatted = f"<p>{intro}</p>" if intro else ""
        for i in range(1, len(steps), 2):
            reformatted += f"<p>{steps[i]} {steps[i+1].strip()}</p>"
        
        return reformatted.strip()
    else:
        # Return plain text as is
        return answer.strip()

def generate_answer(question):

    app = workflow.compile()
    inputs = {"question": question, "max_retries": 1}
    last_value = None
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
            last_value = value
    
    # Check if the output is from query_elasticsearch or generate
    if "documents" in last_value and not "generation" in last_value:
        # Output is from query_elasticsearch
        return last_value["documents"]
    else:
        # Output is from generate
        return last_value["generation"].content

def generate_answer2(question):
    return "This is a placeholder answer for the question: " + question


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

from flask import Flask, request, jsonify
import os
import base64
import io
from PIL import Image
import re

# First define all helper functions
def get_matching_traffic_pairs(folder_path):
    """
    Find matching pairs of in_traffic and out_traffic graphs.
    Returns list of tuples (in_traffic_path, out_traffic_path)
    """
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        # Get all files in the directory
        files = os.listdir(folder_path)
        
        # Dictionary to store pairs
        pairs = {}
        
        for file in files:
            if file.startswith('in_traffic_') or file.startswith('out_traffic_'):
                # Extract the interface name (everything after in_traffic_ or out_traffic_)
                interface_name = file[11:] if file.startswith('in_traffic_') else file[12:]
                
                if interface_name not in pairs:
                    pairs[interface_name] = {'in': None, 'out': None}
                
                if file.startswith('in_traffic_'):
                    pairs[interface_name]['in'] = os.path.join(folder_path, file)
                else:
                    pairs[interface_name]['out'] = os.path.join(folder_path, file)
        
        # Return only complete pairs
        return [(pair['in'], pair['out']) 
                for pair in pairs.values() 
                if pair['in'] is not None and pair['out'] is not None]
                
    except Exception as e:
        print(f"Error in get_matching_traffic_pairs: {str(e)}")
        return []

def create_side_by_side_image(img1_path, img2_path):
    """
    Create a side-by-side image from two images with equal width distribution
    """
    try:
        # Open images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Define target width for the combined image (this will be scaled by CSS)
        target_combined_width = 1600  # Larger base size for better quality
        
        # Calculate the width for each image (equal distribution)
        target_individual_width = target_combined_width // 2
        
        # Calculate scaling ratios while maintaining aspect ratios
        ratio1 = target_individual_width / img1.size[0]
        ratio2 = target_individual_width / img2.size[0]
        
        # Calculate new dimensions
        new_height1 = int(img1.size[1] * ratio1)
        new_height2 = int(img2.size[1] * ratio2)
        
        # Use the larger height for both images
        max_height = max(new_height1, new_height2)
        
        # Resize images
        img1 = img1.resize((target_individual_width, max_height), Image.Resampling.LANCZOS)
        img2 = img2.resize((target_individual_width, max_height), Image.Resampling.LANCZOS)
        
        # Create new image with combined width
        combined_img = Image.new('RGB', (target_combined_width, max_height), (255, 255, 255))
        
        # Paste images side by side
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (target_individual_width, 0))
        
        return combined_img
        
    except Exception as e:
        print(f"Error in create_side_by_side_image: {str(e)}")
        return None

# Define the Flask route handler
@app.route('/ask', methods=['POST'])
# def ask():
#     try:
#         question = request.form.get("question")
        
        
#         return jsonify({"question": question, "answer": answer})
        
#     except Exception as e:
#         return jsonify({
#             "question": question,
#             "answer": f"An error occurred while processing the request: {str(e)}"
#         })

# @app.route("/ask", methods=["POST"])
# # def ask():
# #     question = request.form.get("question")
    
# #     # Create a simple test plot
# #     fig, ax = plt.subplots()
# #     ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
# #     ax.set_title('Test Plot')
    
# #     # Convert plot to base64
# #     img_buffer = io.BytesIO()
# #     fig.savefig(img_buffer, format='png', bbox_inches='tight')
# #     img_buffer.seek(0)
# #     img_str = base64.b64encode(img_buffer.getvalue()).decode()
# #     plt.close(fig)  # Clean up the figure
    
# #     answer = "Here's your test plot!"
# #     formatted_answer = answer + f'\n\n![Test Plot](data:image/png;base64,{img_str})'
    
# #     return jsonify({"question": question, "answer": formatted_answer})

def ask():
    question = request.form.get("question")
    question = question if question.endswith('?') else question + '?'
    print('zzzzzzzzzzzzzzzzzzz')
    print(question)
    answer = generate_answer(question)  # Your existing answer generation function
    
    # Format the answer with proper markdown
    formatted_answer = format_answer(answer)
    
    return jsonify({"question": question, "answer": formatted_answer})

def format_answer(text):
    """Format the answer text with proper markdown formatting"""
    lines = text.split('\n')
    formatted_lines = []
    in_code_block = False
    code_buffer = []
    code_lang = ''
    list_context = None
    list_indent = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Handle numbered lists with code blocks
        list_match = re.match(r'^(\d+)\.\s*(.+)', line)
        if list_match and not in_code_block:
            number = list_match.group(1)
            content = list_match.group(2)
            list_context = number
            list_indent = line.index(number)  # Capture the indentation level
            
            # Check if the content starts a code block
            if content.strip().startswith('```'):
                in_code_block = True
                code_lang = content.strip()[3:].strip()
                code_buffer = []
                if content.strip() != '```' + code_lang:
                    # If there's content after the code block marker
                    remaining_content = content.strip()[3 + len(code_lang):].strip()
                    if remaining_content:
                        code_buffer.append(remaining_content)
            else:
                formatted_lines.append(f"{number}. {content}")
            
        # Handle code blocks
        elif line.strip().startswith('```'):
            if not in_code_block:
                # Start a code block
                in_code_block = True
                code_lang = line.strip()[3:].strip()
                code_buffer = []
                
                # If we're in a list context, add appropriate indentation
                if list_context:
                    formatted_lines.append(' ' * (list_indent + 3) + '```' + code_lang)
            else:
                # End a code block
                if list_context:
                    # Add indentation for code block within list
                    indent = ' ' * (list_indent + 3)
                    formatted_code = '\n'.join(code_buffer)
                    formatted_lines.extend([f"{indent}{line}" for line in formatted_code.split('\n')])
                    formatted_lines.append(indent + '```')
                else:
                    # Regular code block
                    formatted_lines.append('```' + code_lang)
                    formatted_lines.extend(code_buffer)
                    formatted_lines.append('```')
                in_code_block = False
                code_buffer = []
            
        elif in_code_block:
            # Collect lines inside the code block
            code_buffer.append(line)
            
        else:
            # Handle continued text under list items
            if line.strip() and list_context:
                indent = len(line) - len(line.lstrip())
                if indent >= list_indent:
                    formatted_lines.append(' ' * list_indent + line.lstrip())
                else:
                    list_context = None
                    list_indent = 0
                    formatted_lines.append(line)
            else:
                if line.strip():
                    list_context = None
                    list_indent = 0
                formatted_lines.append(line)
        
        i += 1

    # Join lines and clean up
    text = '\n'.join(formatted_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'```\n\n+', '```\n', text)
    text = re.sub(r'\n\n+```', '\n```', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', text)
    
    return text.strip()

def format_code_block(code, lang=''):
    """Format code blocks with proper indentation"""
    code = code.strip()
    
    if lang.lower() in ['yaml', 'yml']:
        try:
            parsed = yaml.safe_load(code)
            formatted = yaml.dump(parsed, default_flow_style=False, indent=2)
            return f"```yaml\n{formatted}\n```"
        except:
            return f"```{lang}\n{code}\n```"
    
    return f"```{lang}\n{code}\n```"



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)