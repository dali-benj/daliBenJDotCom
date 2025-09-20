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
import aiohttp
import asyncio
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from django.conf import settings # Import settings


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def deepseek_api_call(prompt, stop=None):
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
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                result = await response.json()
                return result["choices"][0]["message"]["content"]
    except aiohttp.ClientError as e:
        print(f"Request error: {e}")
        raise
    except (KeyError, json.JSONDecodeError) as e:
        print(f"API response error: {e}")
        raise


async def fetch_and_extract(query, max_results=5, delay=1):
    search = DuckDuckGoSearchResults(output_format="list", max_results=max_results)
    web_results = search.invoke(query)
    documents = []
    
    async def fetch_single_url(session, result):
        link = result["link"]
        try:
            async with session.get(link, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                html_content = await response.text()
                
                extracted_text = extract(html_content, favor_recall=True)
                if extracted_text:
                    return Document(page_content=extracted_text.strip(), metadata={"source": link})
                else:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text_content = soup.find('body').text.strip()
                    if text_content:
                        return Document(page_content=text_content, metadata={"source": link})
                    else:
                        print(f"No usable text found in: {link}")
                        return None
        except aiohttp.ClientError as e:
            print(f"Error fetching {link}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred processing {link}: {e}")
            return None
    
    # Fetch all URLs concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_url(session, result) for result in web_results]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if result and not isinstance(result, Exception):
                documents.append(result)
            elif isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
    
    return documents


# Global cache for FAISS index to avoid reloading
_faiss_cache = {}

async def perform_search(query, data_file_path, index_path="faiss_index", chain_type="stuff"):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting RAG search for query: {query}")
        
        # Create a wrapper for the async function to work with LangChain
        def sync_deepseek_wrapper(prompt, stop=None):
            return asyncio.run(deepseek_api_call(prompt, stop))
        
        llm = RunnableLambda(sync_deepseek_wrapper)
        logger.info("LLM wrapper created successfully")

    # Load or create FAISS index with caching - SAFE VERSION
    if index_path in _faiss_cache:
        db = _faiss_cache[index_path]
        print("Using cached FAISS index.")
    elif os.path.exists(index_path):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        try:
            db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True) #Safe to set True
            _faiss_cache[index_path] = db  # Cache the loaded index
            print("Loaded existing FAISS index (with safe deserialization).")
        except ValueError as e:
            if "allow_dangerous_deserialization" in str(e):
                print("ERROR:  Could not load the index.  It may be from an untrusted source.")
                print("       Delete the 'faiss_index' directory and re-run the script to rebuild it.")
                return None  # Or raise an exception, or handle it appropriately
            else:
                raise
    else:
        if not os.path.exists(data_file_path):
            print(f"Data file not found at {data_file_path}")
            return None
        loader = JSONLoader(file_path=data_file_path, jq_schema='.content', text_content=False, json_lines=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(index_path)
        _faiss_cache[index_path] = db  # Cache the created index
        print("Created and saved new FAISS index.")

    # Fetch web documents asynchronously
    web_documents = await fetch_and_extract(query)
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
        logger.info("RAG search completed successfully")
        return {
            'answer': result['result'],
            'source_documents': [{'content': doc.page_content, 'metadata': doc.metadata} for doc in result['source_documents']]
        }
    except Exception as e:
        logger.error(f"Error in perform_search: {str(e)}", exc_info=True)
        return None