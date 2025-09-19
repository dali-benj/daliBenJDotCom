# Create your views here.
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import sync_to_async
from .forms import QueryForm
from .rag_logic import perform_search
import os
import asyncio
from django.conf import settings
from .models import Query
from django.utils.html import escape


async def rag_view(request):
    results = None
    if request.method == 'POST':
        form = QueryForm(request.POST)
        if form.is_valid():
            # Sanitize the input *before* saving or using it
            query = escape(form.cleaned_data['query'])

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

            # Save query asynchronously
            await sync_to_async(query_obj.save)()

            # Perform async search
            data_file_path = os.path.join(settings.BASE_DIR,  'apps/rag_app/anime_corner_data.jsonl')
            index_path = os.path.join(settings.BASE_DIR, 'faiss_index')
            results = await perform_search(query, data_file_path, index_path)

            if results is None:
                return render(request, 'rag_app/rag.html', {'form': form, 'error_message': "Failed to load RAG system."})
            
            # Save the answer to the database asynchronously
            query_obj.answer_text = results['answer']
            await sync_to_async(query_obj.save)()
            return render(request, 'rag_app/rag.html', {'form': form, 'results': results})

    else:
        form = QueryForm()
    return render(request, 'rag_app/rag.html', {'form': form, 'results': results})

