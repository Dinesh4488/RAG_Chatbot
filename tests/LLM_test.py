from llama_cpp import Llama
import os

# --- CONFIGURATION ---
# UPDATE THIS PATH to exactly where your model is
model_path = r"C:/rag_project/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Check if file exists before crashing
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please check the file name and folder path.")
    exit()

print(f"1. Loading Model from: {model_path}")
print("   (If this freezes for >30s, your RAM might be full)")

try:
    # Initialize the Model
    # n_ctx=512 : Keeps memory usage very low (Safe Mode)
    # n_gpu_layers=-1 : Tries to use your GPU. If it fails, it falls back to CPU.
    llm = Llama(
        model_path=model_path, 
        n_ctx=512, 
        n_gpu_layers=-1, # Change to 0 if you get DLL errors
        verbose=True
    )
    print("✔ Model Loaded Successfully!")

    print("2. Sending Prompt: 'What is the capital of France?'")
    
    # Run Inference
    output = llm(
        "What is the capital of France?", 
        max_tokens=32, # Stop after ~1 sentence
        stop=["Q:", "\n"], 
        echo=True
    )

    print("-" * 30)
    print("3. MODEL RESPONSE:")
    print(output['choices'][0]['text'])
    print("-" * 30)
    print("✔ SUCCESS: No crashes.")

except Exception as e:
    print("\n❌ CRITICAL ERROR:")
    print(e)
    print("\nTROUBLESHOOTING:")
    print("- If 'DLL load failed': Reinstall llama-cpp-python")
    print("- If 'Memory Error': Close Chrome/VS Code and try again")