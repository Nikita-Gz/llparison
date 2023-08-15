# this file tests HF inference API

import json
import requests
import time
import logging
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stablecode-instruct-alpha-3b"

with open("./s/hf_read", 'r') as file:
  API_TOKEN = file.read()

log = logging.getLogger("test.py")
logging.basicConfig(level=logging.INFO)

headers = {"Authorization": f"Bearer {API_TOKEN}"}
def _query_hf(payload, model_name='stabilityai/stablecode-instruct-alpha-3b') -> dict:
  API_URL = 'https://api-inference.huggingface.co/models/' + model_name
  data = json.dumps(payload)
  log.info(f'Sending HF the payload:\n{data[:100]}...')

  while True:
    response = requests.request("POST", API_URL, headers=headers, data=data)
    response_txt = response.content.decode("utf-8")

    # wait a portion of the estimated time if the model is loading
    if response.status_code == 503:
      wait_for = json.loads(response_txt)['estimated_time'] / 2
      log.warning(f'Waiting for {wait_for}')
      time.sleep(wait_for)
    elif response.status_code != 200:
      log.warning(f'Retrying HF request because of bad response ({response.status_code}): {response_txt}')
      time.sleep(1)
    else: # success
      break

  response_json = json.loads(response_txt)
  log.info(f'Got response: {response_json}')
  return response_json


parameters = {
  'temperature': 1,
  'top_p': 0.80,
  'top_k': 50,
  'repetition_penalty': 1.1,
  'return_full_text': False,
  'max_new_tokens': 50,
  'use_cache': False,
  'do_sample': True
}
inputs = "Q: But what am i supposed to do, captain Picard? You're not giving me many choices!\nPicard: "
inputs = """Text:
In the 16th century, an age of great marine and terrestrial exploration, Ferdinand Magellan led the first expedition to sail around the world. As a young Portuguese noble, he served the king of Portugal, but he became involved in the quagmire of political intrigue at court and lost the king’s favor. After he was dismissed from service by the king of Portugal, he offered to serve the future Emperor Charles V of Spain.

A papal decree of 1493 had assigned all land in the New World west of 50 degrees W longitude to Spain and all the land east of that line to Portugal. Magellan offered to prove that the East Indies fell under Spanish authority. On September 20, 1519, Magellan set sail from Spain with five ships. More than a year later, one of these ships was exploring the topography of South America in search of a water route across the continent. This ship sank, but the remaining four ships searched along the southern peninsula of South America. Finally they found the passage they sought near 50 degrees S latitude. Magellan named this passage the Strait of All Saints, but today it is known as the Strait of Magellan.

One ship deserted while in this passage and returned to Spain, so fewer sailors were privileged to gaze at that first panorama of the Pacific Ocean. Those who remained crossed the meridian now known as the International Date Line in the early spring of 1521 after 98 days on the Pacific Ocean. During those long days at sea, many of Magellan’s men died of starvation and disease.

Later, Magellan became involved in an insular conflict in the Philippines and was killed in a tribal battle. Only one ship and 17 sailors under the command of the Basque navigator Elcano survived to complete the westward journey to Spain and thus prove once and for all that the world is round, with no precipice at the edge.
Question: The 16th century was an age of which exploration?
Options:
A) none of the above
B) land
C) biological
D) common man
E) cosmic
The correct answer is the letter: """

data = _query_hf({
  "inputs": inputs,
  'parameters': parameters})

print(data)
print('='*20)
print(data[0]['generated_text'])
