from django.urls import path
from . import views

urlpatterns = [
    path('', views.rag_view, name='rag_index'),  # RAG system at /rag/
]