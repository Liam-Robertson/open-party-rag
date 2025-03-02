# File: open_party_rag/rag_app/data_upload/processed.py
import os
import shutil
PROCESSED_FOLDER = os.path.join(os.path.dirname(__file__), "processed")
def move_file(file_path):
    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)
    destination = os.path.join(PROCESSED_FOLDER, os.path.basename(file_path))
    shutil.move(file_path, destination)
