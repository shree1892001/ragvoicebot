"""
VoiceBot.py
-----------
This module provides the OptimizedRAG class and a command-line interface for an ultra-fast RAG (Retrieval-Augmented Generation) chatbot with voice input/output.

- Uses Ollama for LLM inference (model name and base URL configurable via constants).
- Uses HuggingFace sentence-transformers for embeddings (model name from constants).
- Loads data from a file specified in constants.
- Provides caching, logging, and a warm-up routine.
- Main entry point: main() for interactive voice-based Q&A.
"""
import os
import pickle
import time
import logging
import asyncio
from typing import Optional, Tuple, List
from langchain_community.llms import Ollama
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import concurrent.futures
from functools import lru_cache
from Common.Constants import *
from voice_utils import audio_to_text, text_to_audio

# Setup logging
rag_logger = logging.getLogger("RAGLogger")
rag_logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("../rag_system.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

rag_logger.addHandler(console_handler)
rag_logger.addHandler(file_handler)


class OptimizedRAG:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OptimizedRAG, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("🚀 Initializing Ultra-Fast RAG...")
            rag_logger.info("Initializing OptimizedRAG instance.")
            self.setup_paths()
            self.setup_llm()
            self.setup_embeddings()
            self.load_or_create_vectorstore()
            self.setup_qa_chain()
            self.initialized = True
            print("✅ Ultra-Fast RAG ready!")
            rag_logger.info("OptimizedRAG setup complete.")

    def setup_paths(self):
        self.persist_dir = "../storage_fast"
        self.vectorstore_path = os.path.join(self.persist_dir, "vectorstore.pkl")
        self.data_file = DIR
        os.makedirs(self.persist_dir, exist_ok=True)

    def setup_llm(self):
        try:
            print("🤖 Setting up Ultra-Fast LLM...")
            rag_logger.info("Attempting to connect to Ollama...")
            import requests
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)

            if response.status_code == 200:
                print("✅ Ollama connected")
                rag_logger.info("Ollama connection successful.")
                self.llm = Ollama(
                    model=OLLAMA_MODEL_NAME,
                    base_url=OLLAMA_BASE_URL,
                    temperature=0.0,
                    num_predict=80,
                    top_k=5,
                    top_p=0.8,
                    repeat_penalty=1.05,
                    num_ctx=1024,
                )
                self.llm_available = True
                print("⚡ LLM optimized for maximum speed")
                rag_logger.info("LLM setup complete.")
            else:
                raise Exception("Ollama connection failed")
        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            rag_logger.exception("LLM setup failed")

    def _has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def setup_embeddings(self):
        try:
            print("🧮 Setting up ultra-fast embeddings...")
            rag_logger.info("Setting up HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 16,
                }
            )
            print("✅ Ultra-fast embeddings ready")
            rag_logger.info("Embeddings setup successful.")
        except Exception as e:
            print(f"❌ Embeddings setup failed: {e}")
            rag_logger.warning("Primary embeddings setup failed. Trying fallback.")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME
                )
                print("✅ Basic embeddings ready")
                rag_logger.info("Fallback embeddings setup successful.")
            except Exception as e2:
                print(f"❌ Basic embeddings also failed: {e2}")
                rag_logger.exception("All embeddings setup attempts failed.")
                raise

    def load_or_create_vectorstore(self):
        try:
            if os.path.exists(self.vectorstore_path):
                print("📚 Loading existing vectorstore...")
                rag_logger.info("Loading vectorstore from disk...")
                with open(self.vectorstore_path, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                print("✅ Vectorstore loaded")
                rag_logger.info("Vectorstore loaded.")
                return
            self.create_vectorstore()
        except Exception as e:
            print(f"⚠️ Error loading vectorstore: {e}")
            rag_logger.warning("Vectorstore load failed. Recreating...")
            self.create_vectorstore()

    def create_vectorstore(self):
        try:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file {self.data_file} not found")

            with open(self.data_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            print(f"📄 Processing {len(raw_text)} characters...")
            rag_logger.info("Splitting text into chunks...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )

            texts = text_splitter.split_text(raw_text)
            print(f"📝 Created {len(texts)} optimized chunks")

            documents = [
                Document(page_content=text.strip())
                for text in texts
                if text.strip() and len(text.strip()) > 20
            ]

            print("🗄️ Building optimized vectorstore...")
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            print("✅ Optimized vectorstore created")

            with open(self.vectorstore_path, 'wb') as f:
                pickle.dump(self.vectorstore, f)
            print("💾 Vectorstore saved for future use")
            rag_logger.info("Vectorstore created and saved.")

        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            rag_logger.exception("Failed to create vectorstore.")
            raise

    def setup_qa_chain(self):
        try:
            print("⛓️ Setting up ultra-fast QA chain...")
            rag_logger.info("Setting up QA chain...")

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4, "fetch_k": 8}
            )

            prompt = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
            )
            print("✅ Ultra-fast QA chain ready")
            rag_logger.info("QA chain setup successful.")
        except Exception as e:
            print(f"❌ QA chain setup failed: {e}")
            rag_logger.exception("QA chain setup failed.")
            raise

    @lru_cache(maxsize=100)
    def _cached_query(self, question: str) -> str:
        return self._execute_query(question)

    def _execute_query(self, question: str) -> str:
        try:
            rag_logger.debug(f"Running query: {question}")
            result = self.qa_chain.invoke({"query": question})
            if isinstance(result, dict):
                return result.get('result', str(result)).strip()
            return str(result).strip()
        except Exception as e:
            rag_logger.exception("Query execution failed.")
            raise Exception(f"Query execution failed: {e}")

    def query(self, question: str) -> Tuple[str, int]:
        start_time = time.perf_counter()
        rag_logger.info(f"Received query: {question}")

        try:
            try:
                answer = self._cached_query(question)
                if answer and len(answer) > 10:
                    end_time = time.perf_counter()
                    response_time = int((end_time - start_time) * 1000)
                    print("⚡ Used cached result")
                    rag_logger.info(f"Used cached result. Time: {response_time} ms")
                    return answer, response_time
            except:
                pass

            if not self.llm_available or not hasattr(self, 'qa_chain'):
                raise Exception("LLM or QA chain not ready")

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._execute_query, question)
                answer = future.result(timeout=30)

            if not answer or len(answer) < 5:
                raise Exception("Empty response from LLM")

            end_time = time.perf_counter()
            response_time = int((end_time - start_time) * 1000)
            rag_logger.info(f"Query processed in {response_time} ms")
            return answer, response_time

        except Exception as e:
            end_time = time.perf_counter()
            response_time = int((end_time - start_time) * 1000)
            print(f"❌ Query error: {e}")
            rag_logger.error(f"Query failed: {e}")
            return f"Error: {str(e)}", response_time

    def warm_up(self):
        print("🔥 Warming up system...")
        rag_logger.info("System warm-up started.")
        try:
            _, _ = self.query("What is RedBeryl?")
            print("✅ System warmed up")
            rag_logger.info("System warm-up completed.")
        except:
            print("⚠️ Warm up failed, but system should still work")
            rag_logger.warning("System warm-up failed.")


def main():
    print("🚀 Ultra-Fast RAG Chatbot")
    print("=" * 60)
    rag_logger.info("Chatbot session started.")

    try:
        rag = OptimizedRAG()
        rag.warm_up()

        print("\n💬 Ready! Ask about RedBeryl Technologies")
        print("💡 Type 'exit' to quit")
        print("💡 Type 'clear' to clear cache")
        print("💡 Common questions are cached for instant responses\n")

        while True:
            try:
                question = audio_to_text()

                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    rag_logger.info("User exited the session.")
                    break

                if question.lower() == 'clear':
                    rag._cached_query.cache_clear()
                    print("🗑️ Cache cleared")
                    rag_logger.info("Cache cleared by user.")
                    continue

                if not question:
                    continue

                answer, response_time = rag.query(question)

                # Custom handling for insufficient answers
                def is_price_question(q):
                    keywords = ["price", "cost", "charge", "rate", "pricing", "how much", "fee"]
                    ql = q.lower()
                    return any(k in ql for k in keywords)

                def is_insufficient_answer(ans):
                    if not ans or len(ans.strip()) < 10:
                        return True
                    insufficient_phrases = [
                        "don't have enough information",
                        "not enough information",
                        "cannot answer",
                        "no information",
                        "i do not know",
                        "i don't know",
                        "error:",
                        "empty response"
                    ]
                    ansl = ans.lower()
                    return any(p in ansl for p in insufficient_phrases)

                if is_price_question(question):
                    answer = PRICE_INQUIRY_RESPONSE
                elif is_insufficient_answer(answer):
                    answer = INSUFFICIENT_ANSWER_RESPONSE

                print(f"\n🤖 Answer: {answer}")
                text_to_audio(answer)

                print(f"⏱️  Time: {response_time} ms")
                if response_time < 1000:
                    print("⚡ Lightning fast!")
                elif response_time < 3000:
                    print("🚀 Very fast!")
                elif response_time < 5000:
                    print("✅ Fast!")
                else:
                    print("🐌 Slow - consider optimizing")

                print("-" * 50)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                text_to_audio("Goodbye!")
                rag_logger.info("KeyboardInterrupt - session ended.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                rag_logger.exception("Unexpected error in main loop.")

    except Exception as e:
        print(f"❌ Failed to start: {e}")
        rag_logger.exception("Failed to initialize chatbot.")
        print("\n💡 Troubleshooting tips:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check if tinyllama is installed: ollama pull tinyllama")


if __name__ == "__main__":
    main()
