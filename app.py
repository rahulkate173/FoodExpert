import os
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# --- 1. ENVIRONMENT & PAGE SETUP ---
load_dotenv() # Load local .env file if it exists

st.set_page_config(page_title="Food Text Extractor", layout="centered")
st.title("Food Text Extraction & Tagging")

# --- 2. CONFIGURATION ---
tags_dict = {
    'np': 'nutrition_panel',
    'il': 'ingredient_list',
    'me': 'menu',
    're': 'recipe',
    'fi': 'food_items',
    'di': 'drink_items',
    'fa': 'food_advertistment',
    'fp': 'food_packaging'
}

def create_sample(example):
    return {"role": "user", "content": example}

# --- 3. CACHED MODEL LOADING ---
@st.cache_resource(show_spinner=False)
def load_model():
    BASE_MODEL = "google/gemma-3-270m-it"
    MODEL_NAME = "RahulKate-173/FoodExtract-gemma-3-270m-fine-tune-peft-v2"
    
    # 1. Look in system environment variables (Hugging Face Spaces, Render, Docker)
    hf_token = os.environ.get("HF_TOKEN")
    
    # 2. Fallback to Streamlit Cloud Secrets if not found in environment
    if not hf_token:
        try:
            hf_token = st.secrets["HF_TOKEN"]
        except (KeyError, FileNotFoundError):
            pass

    # If still not found, halt execution and guide the user
    if not hf_token:
        st.error("⚠️ Hugging Face Token (`HF_TOKEN`) not found. Please set it up in your deployment environment variables or Streamlit Secrets.")
        st.stop()

    # Pass the token explicitly to the Hugging Face functions
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL, 
        torch_dtype="auto", 
        device_map="auto",
        attn_implementation="eager",
        token=hf_token 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        token=hf_token
    )
    
    peft_model = PeftModel.from_pretrained(
        base_model, 
        MODEL_NAME,
        token=hf_token
    )
    
    loaded_model_pipeline = pipeline("text-generation", model=peft_model, tokenizer=tokenizer)
    return loaded_model_pipeline

# Display spinner while downloading/loading the model into memory
with st.spinner("Loading model and authenticating with Hugging Face..."):
    try:
        model_pipeline = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- 4. USER INTERFACE ---
text = st.text_area("Enter text to analyze:", height=150, placeholder="Type food related text here...")

if st.button("Generate Tags"):
    if not text.strip():
        st.warning("Please enter some text before generating.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Format the prompt
                prompt = create_sample(text) 
                
                # Apply chat template
                updated_prompt = model_pipeline.tokenizer.apply_chat_template(
                    conversation=[prompt], 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Generate output
                default_outputs = model_pipeline(text_inputs=updated_prompt, max_new_tokens=256)
                
                # Extract the newly generated text 
                output_text = default_outputs[0]["generated_text"][len(updated_prompt):]
                
                # Display results
                st.subheader("Extraction Result:")
                st.info(output_text)
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")