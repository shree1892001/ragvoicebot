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

logging.basicConfig(level=logging.ERROR)


class OptimizedRAG:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OptimizedRAG, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("🚀 Initializing Ultra-Fast RAG...")
            self.setup_paths()
            self.setup_llm()
            self.setup_embeddings()
            self.load_or_create_vectorstore()
            self.setup_qa_chain()
            self.initialized = True
            print("✅ Ultra-Fast RAG ready!")

    def setup_paths(self):
        """Setup storage paths"""
        self.persist_dir = "./storage_fast"
        self.vectorstore_path = os.path.join(self.persist_dir, "vectorstore.pkl")
        self.data_file = "redberyl_tech_all_data.txt"
        os.makedirs(self.persist_dir, exist_ok=True)

    def setup_llm(self):
        """Setup ultra-fast LLM configuration"""
        try:
            print("🤖 Setting up Ultra-Fast LLM...")
            import requests
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)

            if response.status_code == 200:
                print("✅ Ollama connected")
                # Ultra-optimized configuration for speed (using only supported parameters)
                self.llm = Ollama(
                    model="tinyllama",  # Keep fastest model
                    base_url="http://127.0.0.1:11434",
                    temperature=0.0,  # Reduce randomness for faster generation
                    num_predict=80,  # Shorter responses for speed
                    top_k=5,  # Reduce from 10 to 5 for faster sampling
                    top_p=0.8,  # Reduce from 0.9 to 0.8
                    repeat_penalty=1.05,  # Reduce from 1.1 to 1.05
                    num_ctx=1024,  # Reduce context window for speed
                )
                self.llm_available = True
                print("⚡ LLM optimized for maximum speed")
            else:
                print("❌ Ollama connection failed")
                raise Exception("Ollama connection failed")

        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            raise

    def _has_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def setup_embeddings(self):
        """Setup ultra-fast embeddings with optimized settings"""
        try:
            print("🧮 Setting up ultra-fast embeddings...")
            # Use optimized configuration that works across all systems
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 10,  # Smaller batch for faster processing

                }
            )
            print("✅ Ultra-fast embeddings ready")
        except Exception as e:
            print(f"❌ Embeddings setup failed: {e}")
            # Fallback to basic configuration
            try:
                print("🔄 Using basic embeddings configuration...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                print("✅ Basic embeddings ready")
            except Exception as e2:
                print(f"❌ Basic embeddings also failed: {e2}")
                raise

    def load_or_create_vectorstore(self):
        """Load existing vectorstore or create optimized one"""
        try:
            if os.path.exists(self.vectorstore_path):
                print("📚 Loading existing vectorstore...")
                with open(self.vectorstore_path, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                print("✅ Vectorstore loaded")
                return

            print("🔨 Creating optimized vectorstore...")
            self.create_vectorstore()

        except Exception as e:
            print(f"⚠️ Error loading vectorstore: {e}")
            print("🔨 Creating new vectorstore...")
            self.create_vectorstore()

    def create_vectorstore(self):
        """Create optimized vectorstore"""
        try:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file {self.data_file} not found")

            with open(self.data_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            print(f"📄 Processing {len(raw_text)} characters...")

            # Optimized text splitter for better chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Larger chunks to keep lists together
                chunk_overlap=100,  # More overlap to avoid splitting lists
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )

            texts = text_splitter.split_text(raw_text)
            print(f"📝 Created {len(texts)} optimized chunks")

            # Filter out very short chunks
            documents = [
                Document(page_content=text.strip())
                for text in texts
                if text.strip() and len(text.strip()) > 20
            ]

            print("🗄️ Building optimized vectorstore...")
            # Create vectorstore with basic but fast settings
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)

            print("✅ Optimized vectorstore created")

            # Save for future use
            with open(self.vectorstore_path, 'wb') as f:
                pickle.dump(self.vectorstore, f)

            print("💾 Vectorstore saved for future use")

        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise

    def setup_qa_chain(self):
        """Setup ultra-fast QA chain"""
        try:
            print("⛓️ Setting up ultra-fast QA chain...")

            # Ultra-optimized retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 4,  # Increase to 4 for more context
                    "fetch_k": 8  # Increase to 8
                }
            )

            # Improved prompt for better list extraction
            prompt_template = """Context: {context}

Question: {question}

If the question asks for a list or products, extract and provide the full list from the context. Otherwise, answer briefly."""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Create optimized QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
            )

            print("✅ Ultra-fast QA chain ready")

        except Exception as e:
            print(f"❌ QA chain setup failed: {e}")
            raise

    @lru_cache(maxsize=100)
    def _cached_query(self, question: str) -> str:
        """Cached query for repeated questions"""
        return self._execute_query(question)

    def _execute_query(self, question: str) -> str:
        """Execute the actual query"""
        try:
            result = self.qa_chain.invoke({"query": question})

            if isinstance(result, dict):
                answer = result.get('result', str(result))
            else:
                answer = str(result)

            return answer.strip()
        except Exception as e:
            raise Exception(f"Query execution failed: {e}")

    def query(self, question: str) -> Tuple[str, int]:
        """Ultra-fast query processing with optimizations"""
        start_time = time.perf_counter()

        try:
            print(f"🔍 Processing: '{question}'")

            # Check cache first
            try:
                answer = self._cached_query(question)
                if answer and len(answer) > 10:
                    end_time = time.perf_counter()
                    response_time = int((end_time - start_time) * 1000)
                    print("⚡ Used cached result")
                    return answer, response_time
            except:
                pass  # Fall through to regular processing

            # Ensure components are ready
            if not self.llm_available:
                raise Exception("LLM is not available")

            if not hasattr(self, 'qa_chain'):
                raise Exception("QA chain is not initialized")

            # Use timeout for queries
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._execute_query, question)
                try:
                    answer = future.result(timeout=30)  # 30 second timeout
                except concurrent.futures.TimeoutError:
                    raise Exception("Query timed out after 30 seconds")

            if not answer or len(answer) < 5:
                raise Exception("Empty response from LLM")

            end_time = time.perf_counter()
            response_time = int((end_time - start_time) * 1000)

            return answer, response_time

        except Exception as e:
            print(f"❌ Query error: {e}")
            end_time = time.perf_counter()
            response_time = int((end_time - start_time) * 1000)
            return f"Error: {str(e)}", response_time

    def warm_up(self):
        """Warm up the system with a simple query"""
        print("🔥 Warming up system...")
        try:
            _, _ = self.query("What is RedBeryl?")
            print("✅ System warmed up")
        except:
            print("⚠️ Warm up failed, but system should still work")


def main():
    """Main chat loop with optimizations"""
    print("🚀 Ultra-Fast RAG Chatbot")
    print("=" * 60)

    try:
        rag = OptimizedRAG()

        # Warm up the system
        rag.warm_up()

        print("\n💬 Ready! Ask about RedBeryl Technologies")
        print("💡 Type 'exit' to quit")
        print("💡 Type 'clear' to clear cache")
        print("💡 Common questions are cached for instant responses\n")

        while True:
            try:
                question = input("❓ Question: ").strip()

                if question.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break

                if question.lower() == 'clear':
                    rag._cached_query.cache_clear()
                    print("🗑️ Cache cleared")
                    continue

                if not question:
                    continue

                answer, response_time = rag.query(question)

                print(f"\n🤖 Answer: {answer}")
                print(f"⏱️  Time: {response_time} ms")

                # Speed indicator
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
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Check if tinyllama is installed: ollama pull tinyllama")


if __name__ == "__main__":
    main()


