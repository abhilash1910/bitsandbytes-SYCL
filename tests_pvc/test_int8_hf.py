import torch
from transformers import GPTJForCausalLM, AutoTokenizer
import intel_extension_for_pytorch
import accelerate

MAX_NEW_TOKENS = 128

text = "Hamburg is in which country?\n"

tokenizer = AutoTokenizer.from_pretrained('/home/majumder/gpt_j/')
  
input_ids = tokenizer(text, return_tensors="pt").input_ids

max_memory = f"{int(1024**5/1024**3)-2}GB"

n_gpus = torch.xpu.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
model = GPTJForCausalLM.from_pretrained('/home/majumder/gpt_j/', device_map="auto", load_in_8bit=True, max_memory= max_memory)

generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))