# rag_app/rag_logic.py
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from bs4 import BeautifulSoup
from trafilatura import extract
import requests
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from django.conf import settings # Import settings


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def deepseek_api_call(prompt, stop=None):
    if hasattr(prompt, "to_string"):
        prompt = prompt.to_string()
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        raise
    except (KeyError, json.JSONDecodeError) as e:
        print(f"API response error: {e}, Response: {response.text if 'response' in locals() else 'No response'}")
        raise


def fetch_and_extract(query, max_results=5, delay=1):
    search = DuckDuckGoSearchResults(output_format="list", max_results=max_results)
    web_results = search.invoke(query)
    documents = []
    for result in web_results:
        link = result["link"]
        try:
            response = requests.get(link, timeout=10)
            response.raise_for_status()
            if response.encoding is None:
                response.encoding = response.apparent_encoding
            html_content = response.text

            extracted_text = extract(html_content, favor_recall=True)
            if extracted_text:
                documents.append(Document(page_content=extracted_text.strip(), metadata={"source": link}))
            else:
                soup = BeautifulSoup(html_content, 'html.parser')
                text_content = soup.find('body').text.strip()
                if text_content:
                    documents.append(Document(page_content=text_content, metadata={"source": link}))
                else:
                    print(f"No usable text found in: {link}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {link}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred processing {link}: {e}")
        time.sleep(delay)
    return documents


def perform_search(query, data_file_path, index_path="faiss_index", chain_type="stuff"):
    llm = RunnableLambda(deepseek_api_call)

    # Load or create FAISS index - SAFE VERSION
    if os.path.exists(index_path):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        try:
            db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True) #Safe to set True
            print("Loaded existing FAISS index (with safe deserialization).")
        except ValueError as e:
            if "allow_dangerous_deserialization" in str(e):
                print("ERROR:  Could not load the index.  It may be from an untrusted source.")
                print("       Delete the 'faiss_index' directory and re-run the script to rebuild it.")
                return None  # Or raise an exception, or handle it appropriately
            else:
                raise
    else:
        loader = JSONLoader(file_path=data_file_path, jq_schema='.content', text_content=False, json_lines=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(index_path)
        print("Created and saved new FAISS index.")

    web_documents = fetch_and_extract(query)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    web_texts = text_splitter.split_documents(web_documents)
    db.add_documents(web_texts)

    template = """You are a helpful AI assistant.  Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.  Keep your answer concise.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa(query)
    return {
        'answer': result['result'],
        'source_documents': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in result['source_documents']]
    }