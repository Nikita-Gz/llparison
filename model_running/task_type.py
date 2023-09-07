from enum import Enum

class TaskType(Enum):
  READING_COMPREHENSION = 1
  SPAM_CLASSIFICATION = 2
  CUSTOM = 3
  #READING_COMPREHENSION_TEST = 4
task_type_int_to_str = {
  TaskType.READING_COMPREHENSION: 'Reading Comprehension',
  TaskType.SPAM_CLASSIFICATION: 'Spam Classification',
  TaskType.CUSTOM: 'Custom',
  #TaskType.READING_COMPREHENSION_TEST: 'Reading Comprehension Test'
}

