#File: open_party_rag/open_party_rag/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('rag/', include('rag_app.urls')),
]
