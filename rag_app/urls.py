from django.urls import path
from . import views

urlpatterns = [
    path('rag/', views.rag_view, name='rag_index'),  # RAG system at /rag/
    path('', views.home_view, name='home'),  # Homepage at /
]