#File: open_party_rag/rag_app/urls.py
from django.urls import path
from .views import query_rag

urlpatterns = [
    path('query/', query_rag, name='query_rag'),
]
