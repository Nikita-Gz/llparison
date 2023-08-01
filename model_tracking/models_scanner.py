# this file will scan all HuggingFace text generation models, select top ones, and return them as pandas frame

# use this lib: https://huggingface.co/docs/huggingface_hub/v0.5.1/en/index

import json
import requests
import pandas as pd

def get_openrouter_models():
  opentouter_models_link = 'https://openrouter.ai/api/v1/models'
  response = requests.request("GET", opentouter_models_link)
  assert response.status_code == 200, f"OpenRouter models request returned with code {response.status_code}"
  response_json = json.loads(response.content.decode("utf-8"))
  models_list = response_json['data']

  formatted_models = []
  for model in models_list:
    model_data = {
      'openrouter_name': model['id'],
      'owner': model['id'].split('/')[0],
      'name': model['id'].split('/')[1],
      'price_prompt': model['pricing']['prompt'],
      'price_completion': model['pricing']['completion'],
      'context': model['context_length'],
      'prompt_limit': model['input_limits']['prompt_tokens'],
      'max_tokens_limit': model['input_limits']['max_tokens'],
    }
    formatted_models.append(model_data)
  
  df = pd.DataFrame(formatted_models)
  return df

print(get_openrouter_models()[['owner', 'name', 'price_prompt', 'price_completion']])
