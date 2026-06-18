

# Retrieval-Augmented Generation (RAG) Service

This project is a complete, self-contained RAG service built for a software engineering take-home assignment. It can crawl a given website, index its content using vector embeddings, and answer questions based strictly on the information it has collected, complete with source citations and performance timings.

---

## 🏛️ Architecture & Design Decisions

This section details the architecture, design choices, limitations, and potential future improvements, addressing the core "Design Clarity" requirements.

### Architecture
The service is built as a simple Flask API that orchestrates a three-stage RAG pipeline.
1.  **Crawler**: Uses `requests` and `BeautifulSoup` to politely fetch and parse web content, staying within the initial domain.
2.  **Indexer**: Employs `LangChain` to chunk the extracted text. A `sentence-transformers` model then creates vector embeddings for each chunk, which are stored in an in-memory `FAISS` vector index.
3.  **Q&A Service**: The Flask API loads the FAISS index and a local LLM (`TinyLlama`) via the Hugging Face `Transformers` library. When a question is received, the service performs a similarity search on the index, constructs a grounded prompt with the retrieved context, and generates an answer.

### Tradeoffs
* **Local LLM vs. Cloud API**: I chose to run a model locally (`TinyLlama`) instead of using a paid API like OpenAI. **Tradeoff**: This makes the project free to run and fully self-contained but requires more local resources (RAM/CPU) and has a higher initial setup cost due to the model download.
* **FAISS vs. Production Vector DB**: I used `FAISS` for its speed and simplicity as an in-memory index. **Tradeoff**: It's perfect for this project's scale but doesn't persist data as robustly as a production database like Pinecone or Weaviate and is not suitable for multi-user applications.
* **UI: Simple HTML vs. Framework**: The frontend is a single HTML file with vanilla JavaScript. **Tradeoff**: This makes it extremely easy to run (no build step required) but lacks the features and scalability of a modern framework like React or Vue.

### Limitations
* **Content Extraction**: The crawler uses a basic strategy to extract text from `<main>` or `<body>` tags. It will fail on complex, JavaScript-heavy single-page applications (SPAs) and may not effectively remove all boilerplate (like ads or sidebars) from every site.
* **Hardware Dependency**: The performance of the Q&A service is highly dependent on the user's local hardware, specifically CPU speed and available RAM. It can be slow on machines with limited resources.
* **Scalability**: The entire service runs as a single process and loads the model into memory. It is not designed to handle multiple simultaneous users or very large websites (thousands of pages) without significant architectural changes.

### Next Steps
* **Improve Crawler Robustness**: Integrate a more advanced parsing library like `trafilatura` to get cleaner, more consistent main content from diverse web pages and better handle SPA sites.
* **Implement Asynchronous Processing**: For crawling and indexing larger sites, these tasks should be moved to a background worker queue (e.g., Celery) to avoid blocking the API and provide a better user experience.
* **Add Caching**: Implement a simple caching layer (e.g., using Redis) for answers to common questions to reduce redundant processing and improve response times.

---

## ⚙️ Setup and Run Instructions

Follow these steps to set up the environment and run the service.

### 1. Prerequisites
* Python 3.8+
* *(Optional)* A Hugging Face Account (only required if you switch the code to a gated model like Gemma)

### 2. Environment Setup
Clone the repository, create a virtual environment, and install the required dependencies.
```bash
# Clone the repository (if applicable)
# git clone <your-repo-url>
# cd <your-repo-name>

# Create and activate a virtual environment
python -m venv .venv
# On Windows: .\.venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

# Install all required libraries
pip install -r requirements.txt
```

### 3. Hugging Face Authentication (Optional)

Because this project runs the **TinyLlama-1.1B-Chat-v1.0** model by default (which is open and public), **you do not need a Hugging Face token to run it**. You can safely skip this step!

If you switch the backend code to a gated model (like Gemma), you must authenticate:
```bash
# Run the login command and paste your access token when prompted
huggingface-cli login
```

### 4. Run the Service

Start the Flask server. On the first run, the application will download the required AI model, which may take several minutes.

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000`. You can interact with it via the provided `index.html` dashboard, which includes controls to customize:
* **Max Pages**: Control crawling depth.
* **Chunk Size & Overlap**: Tune vector segmentation parameters.
* **Server Status**: Live API connectivity check.
* **Performance Timing Analysis**: View retrieval vs. LLM generation timings.

-----

## 🚀 Evals & Demo

This section provides sample inputs and outputs to demonstrate the functionality of the service.

### API Samples / Evals

Here is an example workflow using `curl` against the live server. The target URL for this example is `http://example.com`.

**1. Crawl Request**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"start_url": "[http://example.com](http://example.com)"}' \
  [http://127.0.0.1:5000/crawl](http://127.0.0.1:5000/crawl)
```

**Crawl Response**

```json
{
  "message": "Crawling completed.",
  "page_count": 1,
  "skipped_count": 0
}
```

-----

**2. Index Request**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{}' \
  [http://127.0.0.1:5000/index](http://127.0.0.1:5000/index)
```

**Index Response**

```json
{
  "message": "Indexing completed.",
  "vector_count": 1
}
```

-----

**3. Ask Request (Answerable Question)**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the purpose of this domain?"}' \
  [http://127.0.0.1:5000/ask](http://127.0.0.1:5000/ask)
```

**Ask Response (Successful)**

```json
{
  "answer": "The purpose of this domain is for use in illustrative examples in documents without prior coordination or asking for permission.",
  "sources": [
    {
      "url": "[http://example.com/](http://example.com/)",
      "snippet": "Example Domain This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission."
    }
  ],
  "timings": { "retrieval_ms": 23, "generation_ms": 1540, "total_ms": 1563 }
}
```

-----

**4. Ask Request (Unanswerable Question)**

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the current weather in Manipal?"}' \
  [http://127.0.0.1:5000/ask](http://127.0.0.1:5000/ask)
```

**Ask Response (Refusal)**

```json
{
  "answer": "I do not have enough information to answer this question.",
  "sources": [...],
  "timings": {...}
}
```


```
