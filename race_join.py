import os
import json

current_dataset = 'RACE'
rc_texts = dict()
rc_questions = dict()
a = 0
for path, _, files in os.walk("./model_tasks/reading_comprehension/RACE/"):
  for file in files:
    a += 1
    print(f'Reading {file}, {a}')
    with open(os.path.join(path, file), 'r') as file:
      file_json = json.load(file)
    text_id = file_json['id']
    text_id = f'{current_dataset}:{text_id}'
    rc_texts[text_id] = file_json['article']
    for question_i, (question, options, answer) in enumerate(zip(file_json['questions'], file_json['options'], file_json['answers'])):
      question_id = f'{text_id}:{question_i}'
      rc_questions[question_id] = {
        'text_id': text_id,
        'question': question,
        'options': options,
        'answer': answer
      }
dataset = {'texts': rc_texts, 'questions': rc_questions}
with open('rc_dataset.txt', 'w') as file:
  json.dump(dataset, file)
print(len(rc_questions))
