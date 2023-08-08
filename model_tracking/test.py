# this file tests HF inference API

import json
import requests
API_URL = "https://api-inference.huggingface.co/models/gpt2"

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
  'max_new_tokens': 1,
  'use_cache': False,
  'do_sample': True
}
inputs = "Q: But what am i supposed to do, captain Picard? You're not giving me many choices!\nPicard: "
inputs = """
Read the following text and answer the question:
A grasshopper spent the summer hopping about in the sun and singing to his heart's content. One day, an ant went hurrying by, looking very hot and weary.
"Why are you working on such a lovely day?" said the grasshopper.
"I'm collecting food for the winter," said the ant, "and I suggest you do the same." And off she went, helping the other ants to carry food to their store. The grasshopper carried on hopping and singing. When winter came the ground was covered with snow. The grasshopper had no food and was hungry. So he went to the ants and asked for food.
"What did you do all summer when we were working to collect our food?" said one of the ants.
"I was busy hopping and singing," said the grasshopper.
"Well," said the ant, "if you hop and sing all summer, and do no work, then you must starve in the winter."

Question: what is the topic of this text? Select the correct answer
A) Importance of planning for the future
B) Comedy
C) Empathy of ants
D) Nothing to do with the future

The correct answer is: 
"""
data = query({
  "inputs": inputs,
  'parameters': parameters})

print(data)
print('='*20)
print(data[0]['generated_text'])
