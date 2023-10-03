from enum import Enum

class TaskType(Enum):
  READING_COMPREHENSION = 1
  #SPAM_CLASSIFICATION = 2
  BOT_DETECTION = 4
  #CUSTOM = 3
  #READING_COMPREHENSION_TEST = 4
task_type_int_to_str = {
  TaskType.READING_COMPREHENSION: 'Reading Comprehension',
  #TaskType.SPAM_CLASSIFICATION: 'Spam Classification',
  #TaskType.CUSTOM: 'Custom',
  TaskType.BOT_DETECTION: 'Bot Detection'
  #TaskType.READING_COMPREHENSION_TEST: 'Reading Comprehension Test'
}

new_tokens_limit_per_task_type = {
  TaskType.READING_COMPREHENSION: 3,
  #TaskType.SPAM_CLASSIFICATION: 'Spam Classification',
  #TaskType.CUSTOM: 'Custom',
  TaskType.BOT_DETECTION: 3
}

