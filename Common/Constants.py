DIR = "E:\\ragbot\\redberyl_tech_all_data.txt"
DATA_FILE = "../redberyl_tech_all_data.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "tinyllama"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
PROMPT_TEMPLATE= """You are a highly accurate and concise assistant. Your task is to answer the question 
                    EXCLUSIVELY using the provided context. DO NOT use any outside knowledge. 
                    If the answer is not explicitly present in the context, state clearly: 
                    I don't have enough information in the provided context to answer that
                    Context: {context}
"""
INSUFFICIENT_ANSWER_RESPONSE = "Sorry, I could not get the answer to your question from the available information."
PRICE_INQUIRY_RESPONSE = "For pricing and commercial details, please contact RedBeryl Technologies directly at their official channels."