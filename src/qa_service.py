# src/qa_service.py
import time
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

class QAService:
    def __init__(self, index_path: str):
        """
        Initializes the Q&A service by loading the vector store and a local LLM from Hugging Face.
        """
        print("Loading vector index...")
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        self.vector_store = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
        
        print("Initializing local LLM from Hugging Face (TinyLlama/TinyLlama-1.1B-Chat-v1.0)...")
        # This may take a few minutes and some disk space on the first run to download the model.
        hf_pipeline = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=50,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            eos_token_id=[2, 13], # Stop on </s> and \n
        )
        
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print("Hugging Face LLM loaded successfully.")

    def ask(self, question: str, top_k: int = 4):
        """
        Answers a question based on the indexed content.
        """
        start_time = time.time()

        # Step 1: Retrieve relevant context
        retrieval_start = time.time()
        retrieved_docs = self.vector_store.similarity_search(question, k=top_k)
        retrieval_end = time.time()

        if not retrieved_docs:
            return {
                "answer": "I do not have enough information to answer this question.",
                "sources": [],
                "timings": {"total_ms": (time.time() - start_time) * 1000}
            }
            
        # Step 2: Construct the grounded prompt
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt_template = """<|system|>
Answer the question in a single short sentence using ONLY the context provided. If the answer is not in the context, say "I do not have enough information."

Context:
{context}</s>
<|user|>
{question}</s>
<|assistant|>
"""
        formatted_prompt = prompt_template.format(context=context, question=question)

        # Step 3: Generate the answer
        generation_start = time.time()
        raw_answer = self.llm.invoke(formatted_prompt)
        generation_end = time.time()
        
        # Clean answer (extract only the assistant's generation, handling LangChain's potential failure to strip prompt on Windows)
        answer = raw_answer
        assistant_tag = "<|assistant|>"
        if assistant_tag in answer:
            answer = answer.split(assistant_tag)[-1].strip()
        else:
            # Fallback manual check
            normalized_prompt = formatted_prompt.replace('\r\n', '\n')
            normalized_answer = answer.replace('\r\n', '\n')
            if normalized_answer.startswith(normalized_prompt):
                answer = normalized_answer[len(normalized_prompt):].strip()
        answer = answer.strip()
        
        # Step 4: Format the response
        sources = []
        for doc in retrieved_docs:
            if doc.metadata['source'] not in [s['url'] for s in sources]:
                sources.append({"url": doc.metadata['source'], "snippet": doc.page_content})

        total_time = time.time() - start_time
        timings = {
            "retrieval_ms": round((retrieval_end - retrieval_start) * 1000),
            "generation_ms": round((generation_end - generation_start) * 1000),
            "total_ms": round(total_time * 1000)
        }

        return {"answer": answer, "sources": sources, "timings": timings}