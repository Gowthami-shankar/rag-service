# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from src.crawler import crawl_site
from src.indexer import build_index
from src.qa_service import QAService

app = Flask(__name__)
CORS(app)

# Define file paths
CRAWLED_DATA_PATH = "data/crawled_content.json"
INDEX_PATH = "data/vector_index"

# Global variable to hold the QA service instance
qa_service = None

def load_qa_service():
    """Loads the QA service if the index exists."""
    global qa_service
    if os.path.exists(INDEX_PATH) and qa_service is None:
        try:
            qa_service = QAService(index_path=INDEX_PATH)
            print("QA Service loaded successfully.")
        except Exception as e:
            print(f"Error loading QA service: {e}")
            qa_service = None
    elif qa_service:
        print("QA Service is already loaded.")
    else:
        print("Index not found. Cannot load QA service.")


@app.route('/crawl', methods=['POST'])
def crawl():
    data = request.json
    start_url = data.get('start_url')
    max_pages = data.get('max_pages', 30)
    
    if not start_url:
        return jsonify({"error": "start_url is required"}), 400

    results = crawl_site(start_url=start_url, max_pages=max_pages)
    
    os.makedirs("data", exist_ok=True)
    with open(CRAWLED_DATA_PATH, 'w') as f:
        json.dump(results['page_content'], f, indent=2)

    return jsonify({
        "message": "Crawling completed.",
        "page_count": results['crawled_count'],
        "skipped_count": results['skipped_count']
    })

@app.route('/index', methods=['POST'])
def index():
    if not os.path.exists(CRAWLED_DATA_PATH):
        return jsonify({"error": "Crawled content not found. Please run /crawl first."}), 404

    with open(CRAWLED_DATA_PATH, 'r') as f:
        crawled_content = json.load(f)

    data = request.json or {}
    chunk_size = data.get('chunk_size', 800)
    chunk_overlap = data.get('chunk_overlap', 100)
    
    vector_count = build_index(crawled_content, chunk_size, chunk_overlap, INDEX_PATH)
    
    load_qa_service()

    return jsonify({
        "message": "Indexing completed.",
        "vector_count": vector_count
    })


@app.route('/ask', methods=['POST'])
def ask():
    global qa_service
    if qa_service is None:
        return jsonify({"error": "QA service not loaded. Please run /index first."}), 503

    data = request.json
    question = data.get('question')
    top_k = data.get('top_k', 4)

    if not question:
        return jsonify({"error": "question is required"}), 400

    response = qa_service.ask(question, top_k)
    return jsonify(response)


if __name__ == '__main__':
    load_qa_service()
    app.run(host='0.0.0.0', port=5000, debug=True)