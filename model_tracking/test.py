# this file tests HF inference API

import json
import requests
API_URL = "https://api-inference.huggingface.co/models/LLaMA-2-7B-32K"

with open("./s/hf_read", 'r') as file:
  API_TOKEN = file.read()

headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
  data = json.dumps(payload)
  response = requests.request("POST", API_URL, headers=headers, data=data)
  return json.loads(response.content.decode("utf-8"))

parameters = {
  'temperature': 1,
  'top_p': 0.92,
  'top_k': 500,
  #'repetition_penalty': 1,
  'return_full_text': False,
  'max_new_tokens': 80,
  'use_cache': False,
  'do_sample': True
}
inputs = "Q: But what am i supposed to do, captain Picard? You're not giving me many choices!\nPicard: "
data = query({
  "inputs": inputs,
  'parameters': parameters})

print(data)
print(data[0]['generated_text'])
