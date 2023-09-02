def get_fake_testing_evaluations():
  return [
    {
      #'_id': uuid4().int,
      'date': '2023-01-02',
      'model_id': 'hf::gpt2',
      'iterations': 2,
      'config': {'temperature': 1.0, 'top-p': 0.5},
      'notes': '',
      'task_type': 'Reading Comprehension',
      'metrics': {'accuracy': 0.65, 'f1': 0.6},
      'outputs': [
        {
          'input_code': 'RACE:high10024.txt:0',
          'interpreted_output': 'A',
          'model_output': '(A)'
        },
        {
          'input_code': 'RACE:high10024.txt:1',
          'interpreted_output': 'B',
          'model_output': '(B)'
        },
        {
          'input_code': 'RACE:high10024.txt:2',
          'interpreted_output': 'C',
          'model_output': '(C)'
        },
      ]
    },
    {
      #'_id': uuid4().int,
      'date': '2023-01-03',
      'model_id': 'hf::gpt2',
      'iterations': 2,
      'config': {'temperature': 1.0, 'top-p': 0.5},
      'notes': '',
      'task_type': 'Reading Comprehension',
      'metrics': {'accuracy': 0.75, 'f1': 0.7},
      'outputs': [
        {
          'input_code': 'RACE:high10024.txt:0',
          'interpreted_output': 'D',
          'model_output': '(D)'
        },
        {
          'input_code': 'RACE:high10024.txt:1',
          'interpreted_output': 'D',
          'model_output': 'D'
        },
        {
          'input_code': 'RACE:high10024.txt:2',
          'interpreted_output': 'C',
          'model_output': '(C)'
        },
      ]
    },
    {
      #'_id': uuid4().int,
      'date': '2023-01-02',
      'model_id': 'hf::gpt2',
      'iterations': 2,
      'config': {'temperature': 0.01, 'top-p': 0.5},
      'notes': '',
      'task_type': 'Reading Comprehension',
      'metrics': {'accuracy': 0.75, 'f1': 0.7},
    },
    {
      #'_id': uuid4().int,
      'date': '2023-01-02',
      'model_id': 'hf::gpt2',
      'iterations': 2,
      'config': {'temperature': 0.01, 'top-p': 0.5, 'testparam': 1},
      'notes': '',
      'task_type': 'Reading Comprehension',
      'metrics': {'accuracy': 0.75, 'f1': 0.7},
      'outputs': [
        {
          'input_code': 1,
          'interpreted_output': 'B',
          'model_output': '(BBB)'
        },
        {
          'input_code': 2,
          'interpreted_output': 'B',
          'model_output': '(BBBBBB)'
        },
        {
          'input_code': 3,
          'interpreted_output': 'A',
          'model_output': '(AAAa)'
        },
      ]
    },
    {
      #'_id': uuid4().int,
      'date': '2023-01-02',
      'model_id': 'hf::idkanymoreplshelp',
      'iterations': 2,
      'config': {'temperature': 1},
      'notes': '',
      'task_type': 'Reading Comprehension',
      'metrics': {'accuracy': 0.75, 'f1': 0.7},
    },
  ]
