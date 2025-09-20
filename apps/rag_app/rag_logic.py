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


def sync_deepseek_api_call(prompt, stop=None):
    """Synchronous version of DeepSeek API call for use with LangChain."""
    import requests
    import json
    from django.conf import settings
    
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
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Request error: {e}")
        raise
    except (KeyError, json.JSONDecodeError) as e:
        print(f"API response error: {e}")
        raise


async def fetch_searxng_results(query, max_results=3):
    """Fetch search results from self-hosted SearxNG instance."""
    import aiohttp
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # SearxNG API endpoint
        searxng_url = "http://searxng:8080/search"
        params = {
            'q': query,
            'format': 'json',
            'categories': 'general',
            'safesearch': '0'
        }
        
        # Add User-Agent header to avoid 403 errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; RAG-System/1.0)'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(searxng_url, params=params, headers=headers, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for result in data.get('results', [])[:max_results]:
                        results.append({
                            'link': result.get('url', ''),
                            'title': result.get('title', ''),
                            'snippet': result.get('content', '')
                        })
                    
                    logger.info(f"SearxNG returned {len(results)} results")
                    return results
                else:
                    response_text = await response.text()
                    logger.error(f"SearxNG returned status {response.status}: {response_text[:200]}")
                    return []
                    
    except Exception as e:
        logger.error(f"SearxNG search error: {str(e)}")
        return []

async def fetch_and_extract(query, max_results=3, delay=2):
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    # Check cache first to avoid duplicate API calls
    cache_key = f"{query.lower().strip()}_{max_results}"
    if cache_key in _search_cache:
        logger.info(f"Using cached search results for '{query}'")
        return _search_cache[cache_key]
    
    # Try DuckDuckGo first, then fallback to SearxNG
    web_results = []
    
    # First attempt: DuckDuckGo
    logger.info(f"Trying DuckDuckGo search for '{query}' with max_results={max_results}")
    await asyncio.sleep(delay)  # Initial delay
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            search = DuckDuckGoSearchResults(
                output_format="list", 
                max_results=max_results,
                safesearch="moderate"
            )
            web_results = search.invoke(query)
            logger.info(f"DuckDuckGo search successful")
            break
        except Exception as e:
            if "Ratelimit" in str(e) or "202" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (3 ** attempt) + 2
                    logger.warning(f"DuckDuckGo rate limited, waiting {wait_time} seconds before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"DuckDuckGo failed after {max_retries} attempts, trying SearxNG fallback")
                    # Fallback to SearxNG
                    web_results = await fetch_searxng_results(query, max_results)
                    if not web_results:
                        raise Exception(f"Both DuckDuckGo and SearxNG failed: {str(e)}")
                    break
            else:
                logger.error(f"DuckDuckGo search error: {str(e)}")
                raise Exception(f"Web search error: {str(e)}")
    
    if not web_results:
        logger.warning("No web search results found from any source")
        raise Exception("No web search results found")
    
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
    
    # Fetch all URLs with staggered requests to be more respectful
    async with aiohttp.ClientSession() as session:
        results = []
        for i, result in enumerate(web_results):
            # Add small delay between requests to avoid overwhelming servers
            if i > 0:
                await asyncio.sleep(0.5)
            doc = await fetch_single_url(session, result)
            results.append(doc)
        
        for result in results:
            if result and not isinstance(result, Exception):
                documents.append(result)
            elif isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
    
    # Cache the results to avoid future duplicate API calls
    _search_cache[cache_key] = documents
    logger.info(f"Cached search results for '{query}' ({len(documents)} documents)")
    
    return documents


# Global cache for FAISS index to avoid reloading
_faiss_cache = {}

# Simple cache for search results to avoid duplicate API calls
_search_cache = {}

async def perform_search(query, data_file_path, index_path="faiss_index", chain_type="stuff"):
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting RAG search for query: {query}")
        
        # Use the synchronous version directly
        llm = RunnableLambda(sync_deepseek_api_call)
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
                
                # Update metadata for existing documents to show just filename
                import os
                filename = os.path.basename(data_file_path)
                # Note: FAISS doesn't allow direct metadata updates, but new documents will have correct metadata
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
            
            # Update metadata to show just the filename
            import os
            filename = os.path.basename(data_file_path)
            for text in texts:
                text.metadata['source'] = filename
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(index_path)
            _faiss_cache[index_path] = db  # Cache the created index
            print("Created and saved new FAISS index.")

        # Fetch web documents asynchronously
        web_documents = await fetch_and_extract(query)
        if not web_documents:
            logger.error("Failed to fetch web documents - this is required for the RAG system")
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        web_texts = text_splitter.split_documents(web_documents)
        db.add_documents(web_texts)
        logger.info(f"Added {len(web_texts)} web documents to the vector store")

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
        
        # Debug: Log the source documents
        logger.info(f"Number of source documents from QA: {len(result['source_documents'])}")
        for i, doc in enumerate(result['source_documents']):
            logger.info(f"Source doc {i}: content length={len(doc.page_content)}, metadata={doc.metadata}")
            # Log first 100 characters of content for debugging
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.info(f"Source doc {i} content preview: {content_preview}")
        
        source_docs = [{'content': doc.page_content, 'metadata': doc.metadata} for doc in result['source_documents']]
        logger.info(f"Processed source docs: {len(source_docs)}")
        
        return {
            'answer': result['result'],
            'source_documents': source_docs
        }
    except Exception as e:
        logger.error(f"Error in perform_search: {str(e)}", exc_info=True)
        return None