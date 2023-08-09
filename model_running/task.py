from enum import Enum
import json
import os
from typing import *
import string

from model_runner import ModelRunner

# todo: !! custom tasks must be saved to db and loaded from it

class TaskType(Enum):
  READING_COMPREHENSION = 1
  SPAM_CLASSIFICATION = 2
  CUSTOM = 3

class TaskOutput:
  def __init__(self) -> None:
    self.metrics = None
    self.model_outputs = None
    self.interpreted_outputs = None
    self.input_codes = None # input code will allow Task class to return the exact input

class Task:
  alphabet2idx = {i:letter for i, letter in enumerate(string.ascii_uppercase)}
  idx2alphabet = {letter:i for i, letter in enumerate(string.ascii_uppercase)}

  def __init__(self, task_type: TaskType) -> None:
    self.type = task_type
  
  def load_reading_comprehension_data(self) -> Tuple[Dict, Dict]:
    # loads RACE data.
    # Stores it as a dict of texts {text_id: text} and questions {question_id: {text_id, question, options, answer}}
    current_dataset = 'RACE'
    rc_texts = dict()
    rc_questions = dict()
    for path, _, files in os.walk("../model_tasks/reading_comprehension/RACE/"):
      for file in files:
        with open(os.path.join(path, file), 'r') as file:
          file_json = json.load(file)
        text_id = file_json['id']
        rc_texts['text_id'] = f'{current_dataset}:{text_id}'
        for question_i, (question, options, answer) in enumerate(zip(file_json['questions'], file_json['options'], file_json['answers'])):
          question_id = f'{current_dataset}:{text_id}:{question_i}'
          rc_questions[question_id] = {
            'text_id': text_id,
            'question': question,
            'options': options,
            'answer': answer
          }
    
    return rc_texts, rc_questions

  def run_reading_comprehension(self, model, config):
    rc_texts, rc_questions = self.load_reading_comprehension_data()

    # todo: make the preprocessing code for the specific model on a specific task
    # creates a single prompt from the text and question
    def prompt_constructor(text, question_dict):
      question_text = question_dict['question']
      options_text = ''
      for i, option_text in enumerate(question_dict['options']):
        letter = self.idx2alphabet[i]
        options_text += f'{letter}) {option_text}\n'
      options_text = options_text.strip()
      final_text = '\n'.join(['Text:', text, 'Question:', question_text, 'Options:', options_text, 'Answer:'])
      return final_text

    runner = ModelRunner(model, config)
    model_outputs = []
    for question_id in rc_questions:
      question_dict = rc_questions[question_id]
      text = rc_texts[question_dict['text_id']]
      prompt = prompt_constructor(text, question_dict)
      model_outputs.append(runner.model_run_function(prompt))
    
    # todo: continue from here?

  # returns a list of metrics, outputs
  def run_task(self):
    pass



