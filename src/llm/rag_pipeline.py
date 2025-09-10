import os
import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("models/faiss_index.index")

# Load metadata
with open("models/review_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Initialize OpenAI
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.2, model="gpt-3.5-turbo")

def query_rag_system(user_query, top_k=3):
    # Embed query
    query_vector = embedding_model.encode([user_query])
    
    # Search similar reviews
    distances, indices = index.search(np.array(query_vector), top_k)
    matched_reviews = [metadata[i] for i in indices[0]]

    # Prepare context
    context = "\n\n".join([f"Review: {item['review_text']} (Sentiment: {item['sentiment']})" for item in matched_reviews])
    
    # Prompt
    prompt = f"""
You are a helpful retail assistant. Use the reviews below to answer the question.
Reviews:
{context}

Question: {user_query}
Answer:"""

    response = llm.predict(prompt)
    return response
