#File: open_party_rag/open_party_rag/wsgi.py
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'open_party_rag.settings')
application = get_wsgi_application()
