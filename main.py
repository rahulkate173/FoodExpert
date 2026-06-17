from transformers import AutoModelForCausalLM , AutoTokenizer , pipeline 
from peft import PeftModel
from dotenv import load_dotenv
load_dotenv()
# Our fine-tuned model will assign tags to text so we can easily filter them by type in the future
tags_dict = {'np': 'nutrition_panel',
 'il': 'ingredient_list',
 'me': 'menu',
 're': 'recipe',
 'fi': 'food_items',
 'di': 'drink_items',
 'fa': 'food_advertistment',
 'fp': 'food_packaging'}

## creata easy sample 
def create_sample(example):
  return {"role":"user","content":example}

BASE_MODEL = "google/gemma-3-270m-it"
MODEL_NAME = "RahulKate-173/FoodExtract-gemma-3-270m-fine-tune-peft-v2"
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = BASE_MODEL , 
    dtype="auto",
    device_map = "auto",
    attn_implementation = "eager",
)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = MODEL_NAME)
peft_model = PeftModel.from_pretrained(base_model,MODEL_NAME) # here the base mdoel is converted to peft model
## model and tokenizer loading complete

## enter the text 
text = input("Enter text :\n")

## creating pipeline 
from transformers import pipeline 
loaded_model_pipeline = pipeline("text-generation",model=peft_model,tokenizer=tokenizer)
prompt = create_sample(text) # prompt format 
updated_prompt = loaded_model_pipeline.tokenizer.apply_chat_template(conversation = [prompt] ,tokenize=False,add_generation_prompt=True)
default_outputs = loaded_model_pipeline(text_inputs = updated_prompt,max_new_tokens = 256)
print(f"[INFO] Input :\n {text}")
print("[INFO] Outputs generated!!")
print(f"[INFO] Output :\n{default_outputs[0]["generated_text"][len(updated_prompt):]}")