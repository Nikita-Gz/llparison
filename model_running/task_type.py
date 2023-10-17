from enum import Enum

class TaskType(Enum):
  READING_COMPREHENSION = 1
  BASIC_MULTIPLICATION = 2
  BOT_DETECTION = 4

task_type_int_to_str = {
  TaskType.READING_COMPREHENSION: 'Reading Comprehension',
  TaskType.BOT_DETECTION: 'Bot Detection',
  TaskType.BASIC_MULTIPLICATION: 'Basic Math'
}
task_type_str_to_int = {s:i for i, s in task_type_int_to_str.items()}

new_tokens_limit_per_task_type_int = {
  TaskType.READING_COMPREHENSION: 3,
  TaskType.BOT_DETECTION: 1,
  TaskType.BASIC_MULTIPLICATION: 5
}

new_tokens_limit_per_task_type_str = {s:new_tokens_limit_per_task_type_int[i] for i, s in task_type_int_to_str.items()}

