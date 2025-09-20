# Create your views here.
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .forms_simple import SimpleQueryForm
import os
import logging
import requests
import json
from django.conf import settings
from .models import Query
from django.utils.html import escape

logger = logging.getLogger(__name__)

def simple_deepseek_api_call(prompt, stop=None):
    """Simple synchronous API call to DeepSeek"""
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
        logger.info("Making API call to DeepSeek...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        logger.info("API call successful")
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        raise
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"API response error: {e}")
        raise

def fetch_web_results(query, max_results=3):
    """Fetch web search results using DuckDuckGo"""
    try:
        from duckduckgo_search import DDGS
        
        logger.info(f"Searching web for: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        logger.info(f"Found {len(results)} web results")
        return results
    except Exception as e:
        logger.error(f"Error fetching web results: {e}")
        return []

def simple_perform_search(query, data_file_path, index_path="faiss_index"):
    """Simple version of RAG search with web results"""
    try:
        logger.info(f"Starting simple RAG search for query: {query}")
        
        # Fetch web search results
        web_results = fetch_web_results(query, max_results=3)
        
        # Build context from web results
        context_parts = []
        source_documents = []
        
        for i, result in enumerate(web_results):
            title = result.get('title', 'Unknown Title')
            body = result.get('body', 'No content available')
            url = result.get('href', 'Unknown URL')
            
            context_parts.append(f"Source {i+1}: {title}\nContent: {body}")
            source_documents.append({
                'content': f"{title}: {body}",
                'metadata': {'source': url, 'title': title}
            })
        
        # Add local data if available
        if os.path.exists(data_file_path):
            try:
                with open(data_file_path, 'r', encoding='utf-8') as f:
                    local_content = f.read()
                context_parts.append(f"Local Knowledge Base:\n{local_content}")
                source_documents.append({
                    'content': local_content,
                    'metadata': {'source': 'local_knowledge_base'}
                })
            except Exception as e:
                logger.warning(f"Could not read local data file: {e}")
        
        # Combine all context
        full_context = "\n\n".join(context_parts) if context_parts else "No context available"
        
        # Create the prompt with real context
        prompt = f"""Based on the following context from web search results and knowledge base, answer the question: {query}

Context:
{full_context}

Question: {query}

Please provide a helpful answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
        
        try:
            # Check if API key is available
            if not hasattr(settings, 'DEEPSEEK_API_KEY') or not settings.DEEPSEEK_API_KEY:
                logger.warning("No DeepSeek API key found, using fallback response")
                return {
                    'answer': f"Hello! You asked: '{query}'. I found {len(web_results)} web results but need an API key to process them. Please configure the DeepSeek API key.",
                    'source_documents': source_documents
                }
            
            answer = simple_deepseek_api_call(prompt)
            logger.info("Simple search with web results completed successfully")
            return {
                'answer': answer,
                'source_documents': source_documents
            }
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {
                'answer': f"I'm sorry, I couldn't process your request due to an API error: {str(e)}. However, I found {len(web_results)} web results that might be relevant.",
                'source_documents': source_documents
            }
            
    except Exception as e:
        logger.error(f"Error in simple_perform_search: {str(e)}", exc_info=True)
        return None

def rag_view(request):
    logger.info(f"RAG view called with method: {request.method}")
    results = None
    if request.method == 'POST':
        logger.info("POST request received")
        form = SimpleQueryForm(request.POST)
        logger.info(f"Form is valid: {form.is_valid()}")
        if not form.is_valid():
            logger.error(f"Form errors: {form.errors}")
        if form.is_valid():
            try:
                # Sanitize the input *before* saving or using it
                query = escape(form.cleaned_data['query'])
                logger.info(f"Processing RAG query: {query}")

                # Store the query in the database
                query_obj = Query(query_text=query)

                # Get and store IP address and User-Agent
                x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
                if x_forwarded_for:
                    ip = x_forwarded_for.split(',')[0]
                else:
                    ip = request.META.get('REMOTE_ADDR')
                query_obj.ip_address = ip

                user_agent = request.META.get('HTTP_USER_AGENT')
                query_obj.user_agent = user_agent[:255] if user_agent else None # Truncate to fit.

                # Save query
                query_obj.save()
                logger.info("Query saved to database successfully")

                # Perform search using asyncio.run
                data_file_path = os.path.join(settings.BASE_DIR,  'apps/rag_app/anime_corner_data.jsonl')
                index_path = os.path.join(settings.BASE_DIR, 'faiss_index')
                
                logger.info(f"Data file path: {data_file_path}")
                logger.info(f"Index path: {index_path}")
                logger.info(f"Data file exists: {os.path.exists(data_file_path)}")
                logger.info(f"Index exists: {os.path.exists(index_path)}")
                
                # Use simple search function
                results = simple_perform_search(query, data_file_path, index_path)
                logger.info(f"Search results: {results is not None}")

                if results is None:
                    error_msg = "Failed to load RAG system. Check logs for details."
                    logger.error("RAG search returned None")
                    return render(request, 'rag_app/rag.html', {'form': form, 'error_message': error_msg})
                
                # Save the answer to the database
                query_obj.answer_text = results['answer']
                query_obj.save()
                logger.info("Answer saved to database successfully")
                return render(request, 'rag_app/rag.html', {'form': form, 'results': results})
                
            except Exception as e:
                logger.error(f"Error in RAG view: {str(e)}", exc_info=True)
                return render(request, 'rag_app/rag.html', {'form': form, 'error_message': f"An error occurred: {str(e)}"})

    else:
        form = SimpleQueryForm()
    return render(request, 'rag_app/rag.html', {'form': form, 'results': results})