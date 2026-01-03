import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import config

def get_qwen_llm():
    """
    Initializes the Qwen 2.5 0.5B Instruct model pipeline.
    """
    print(f"Loading LLM: {config.LLM_MODEL_ID}...")
    
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    print("Loading Model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID,
        torch_dtype=torch.float32, # Use float16 if you have a GPU, float32 for CPU safety
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.15
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    print("LLM initialization complete.")
    return llm