# this file tests HF API

from huggingface_hub import list_models, model_info, ModelFilter

with open("./s/hf_read", 'r') as file:
  API_TOKEN = file.read()

filter = ModelFilter(task='text-generation')

models = list_models(token=API_TOKEN, sort='downloads', filter=filter, direction=-1, full=True)
download_counts = []
for model in models:
  download_counts.append(model.downloads)

download_counts = [c for c in download_counts if c > 0]

print(sum(download_counts) / len(download_counts))
print(max(download_counts))
