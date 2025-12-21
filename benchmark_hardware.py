import time
import torch
import psutil
import requests
import gc
import sys
from sentence_transformers import CrossEncoder, SentenceTransformer

# ==========================================
# ⚙️ USER CONFIGURATION
# ==========================================
MODEL_EMBEDDING_NAME = "BAAI/bge-m3"
MODEL_RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
OLLAMA_MODEL_NAME = "qwen3:8b"
# 👇 CHANGED: Pointing to the chat endpoint
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Test Data
TEST_QUERY = "What are the specific conditions for financial aid?"
TEST_DOC_TEXT = """
Financial aid is granted to persons whose income and assets are insufficient to cover their maintenance.
The amount is calculated based on the difference between recognized expenses and allowable income.
Recognized expenses include a basic maintenance allowance, rent up to a maximum limit, and health insurance premiums.
Income includes all net earnings, social security benefits, and potential income from assets.
"""
BATCH_SIZE = 50
TEST_DOCS = [TEST_DOC_TEXT for _ in range(BATCH_SIZE)]

# ==========================================
# 🔧 HELPER FUNCTIONS
# ==========================================

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def benchmark_model(model_class, model_name, device, task_name):
    print(f"   👉 Testing on {device.upper()}...", end="", flush=True)
    try:
        clear_vram()
        # 1. Load
        start_load = time.time()
        if model_class == SentenceTransformer:
            model = model_class(model_name, device=device)
        else:
            model = model_class(model_name, device=device)
        load_time = time.time() - start_load

        # 2. Warmup
        if model_class == SentenceTransformer:
            model.encode("warmup", convert_to_tensor=True)
        else:
            model.predict([("warmup", "warmup")])

        # 3. Benchmark Run
        start_run = time.time()
        if model_class == SentenceTransformer:
            model.encode(TEST_DOCS, batch_size=32, convert_to_tensor=True)
        else:
            pairs = [(TEST_QUERY, doc) for doc in TEST_DOCS]
            model.predict(pairs, batch_size=32)
        
        duration = time.time() - start_run
        ms_per_doc = (duration * 1000) / BATCH_SIZE
        print(f" DONE. (Load: {load_time:.2f}s | Run: {duration:.2f}s | Speed: {ms_per_doc:.2f} ms/doc)")
        
        del model
        return duration, ms_per_doc

    except Exception as e:
        print(f" FAILED. ({str(e)})")
        return float('inf'), float('inf')

def test_ollama_chat():
    """Checks Ollama speed using the CHAT API."""
    print(f"\n[3] Testing Ollama Chat ({OLLAMA_MODEL_NAME})...", end="", flush=True)
    try:
        start = time.time()
        
        # 👇 CHANGED: Using 'messages' format for Chat API
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Write a 50 word summary of the history of AI."}
            ],
            "stream": False
        }
        
        res = requests.post(OLLAMA_API_URL, json=payload).json()
        
        # Ollama returns metrics at the root level for both endpoints
        eval_count = res.get("eval_count", 0)
        eval_duration = res.get("eval_duration", 1) # nanoseconds
        
        # Calculate Tokens Per Second
        if eval_duration > 0:
            eval_tps = eval_count / (eval_duration / 1e9)
        else:
            eval_tps = 0
            
        print(f" DONE. ({eval_tps:.2f} tokens/sec)")
        return eval_tps
    except Exception as e:
        print(f" FAILED. Is Ollama running? ({e})")
        return 0

# ==========================================
# 🏁 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"🏎️  RAG COMPONENT SPEED TEST (Chat API Version)")
    print(f"==================================================")
    print(f"HARDWARE DETECTED:")
    print(f" - CPU: {psutil.cpu_count(logical=False)} Cores")
    print(f" - GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
    print(f"==================================================\n")

    results = {}

    # --- ROUND 1: EMBEDDING ---
    print("--- ROUND 1: Embedding (BGE-M3) ---")
    cpu_time_emb, cpu_speed_emb = benchmark_model(SentenceTransformer, MODEL_EMBEDDING_NAME, "cpu", "Embedding")
    
    if torch.cuda.is_available():
        gpu_time_emb, gpu_speed_emb = benchmark_model(SentenceTransformer, MODEL_EMBEDDING_NAME, "cuda", "Embedding")
    else:
        gpu_time_emb, gpu_speed_emb = float('inf'), float('inf')

    # --- ROUND 2: RERANKER ---
    print("\n--- ROUND 2: Reranker (BGE-Reranker) ---")
    cpu_time_rerank, cpu_speed_rerank = benchmark_model(CrossEncoder, MODEL_RERANKER_NAME, "cpu", "Reranker")
    
    if torch.cuda.is_available():
        gpu_time_rerank, gpu_speed_rerank = benchmark_model(CrossEncoder, MODEL_RERANKER_NAME, "cuda", "Reranker")
    else:
        gpu_time_rerank, gpu_speed_rerank = float('inf'), float('inf')

    # --- ROUND 3: OLLAMA CHAT ---
    test_ollama_chat()

    # --- ROUND 4: CAPACITY CHECK ---
    print("\n--- ROUND 4: NVIDIA Capacity Stress Test ---")
    if torch.cuda.is_available():
        try:
            print("Attempting to load BOTH models into VRAM...", end="")
            clear_vram()
            m1 = SentenceTransformer(MODEL_EMBEDDING_NAME, device="cuda")
            print(f" Emb Loaded ({get_vram_usage():.2f}GB)...", end="")
            m2 = CrossEncoder(MODEL_RERANKER_NAME, device="cuda")
            print(f" Reranker Loaded ({get_vram_usage():.2f}GB)...", end="")
            print(" ✅ FITS!")
            fits_in_vram = True
        except Exception as e:
            print(f" ❌ CRASHED/OOM ({e})")
            fits_in_vram = False
    else:
        fits_in_vram = False

    # ==========================================
    # 🏆 FINAL VERDICT
    # ==========================================
    print("\n" + "="*50)
    print("🏆  FINAL CONFIGURATION RECOMMENDATION")
    print("="*50)

    # 1. Embedding Decision
    emb_device = "cuda" if (gpu_time_emb < cpu_time_emb and fits_in_vram) else "cpu"
    
    # 2. Reranker Decision
    rerank_device = "cuda" if (gpu_time_rerank < cpu_time_rerank and fits_in_vram) else "cpu"
    
    print(f"Based on your hardware speeds:")
    print(f"1. EMBEDDING MODEL: [{emb_device.upper()}]")
    print(f"2. RERANKER MODEL:  [{rerank_device.upper()}]")
    print(f"3. OLLAMA (CHAT):   [GPU/CUDA]")
    
    print("\n👇 COPY THIS INTO YOUR CONFIG 👇")
    print(f"device_embedding = '{emb_device}'")
    print(f"device_reranker  = '{rerank_device}'")
    print("="*50)