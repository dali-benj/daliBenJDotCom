# Create your views here.
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .forms import QueryForm
from .rag_logic_simple import simple_perform_search
import os
import logging
from django.conf import settings
from .models import Query
from django.utils.html import escape

logger = logging.getLogger(__name__)

def rag_view(request):
    results = None
    if request.method == 'POST':
        form = QueryForm(request.POST)
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
        form = QueryForm()
    return render(request, 'rag_app/rag.html', {'form': form, 'results': results})