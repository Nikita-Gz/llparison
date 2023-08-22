def get_fake_experiment_result_dicts():
  experiment_dicts = [
    {
      'date': '01072023',
      'model_id': 'model1_id',
      'iterations': 1,
      'config': {'temperature': 1.0, 'top-p': 0.5},
      'notes': '',
      'task_type': 'Reading Comprehension',
      'metrics': {'accuracy': 0.6},
      'model_outputs': ['(A)', 'B', 'B', 'A'],
      'interpreted_outputs': [0, 1, 1, 0],
      'input_codes': [1, 2, 3, 4],
    },
    {
      'date': '01072023',
      'model_id': 'model1_id',
      'iterations': 1,
      'config': {'temperature': 1.0, 'top-p': 0.5},
      'notes': '',
      'task_type': 'Reading Comprehension',
      'metrics': {'accuracy': 0.8},
      'model_outputs': ['(A)', 'C', 'B', 'A'],
      'interpreted_outputs': [0, 2, 1, 0],
      'input_codes': [1, 2, 3, 4],
    },
    {
      'date': '01072023',
      'model_id': 'model1_id',
      'iterations': 1,
      'config': {'temperature': 1.0, 'top-p': 0.5},
      'notes': '',
      'task_type': 'Classification',
      'metrics': {'accuracy': 0.8},
      'model_outputs': ['T', 'F', 'F', 'T'],
      'interpreted_outputs': [1, 0, 0, 1],
      'input_codes': [1, 2, 3, 4],
    },
  ]
  return experiment_dicts


def get_processed_dict_for_output():
  general_data = {
    'models_list': ['model1_id', 'model2_id'],
    'selected_model': 'model1_id'
  }
  models_data = {
    'model1_id': {
      'evaluation_error_message': '',
      'filters':
      [
        {
          'name': 'temperature',
          'values':
          [
            1.0, 0.5, 0.01
          ],
          'default': 1.0
        },
        {
          'name': 'top-p',
          'values':
          [
            0.2, 0.5, 0.9
          ],
          'default': 0.5
        },
      ],
      'task_ratings':
      [
        {
          'task_name': 'Reading Comprehension',
          'metrics':
          [
            {
              'name': 'Accuracy',
              'value': 0.34
            },
            {
              'name': 'False Positives',
              'value': 0.78
            }
          ]
        },
        {
          'task_name': 'Spam Detection',
          'metrics':
          [
            {
              'name': 'Accuracy',
              'value': 0.7
            },
            {
              'name': 'F1',
              'value': 0.6
            }
          ]
        }
      ]
    },

    'model2_id': {
      'evaluation_error_message': 'Ueueueue',
      'filters':
      [
        {
          'name': 'temperature',
          'values':
          [
            1.0, 0.01
          ],
          'default': 0.01
        },
        {
          'name': 'top-p',
          'values':
          [
            0.2, 0.9
          ],
          'default': 0.9
        },
      ],
      'task_ratings':
      [
        {
          'task_name': 'Idk',
          'metrics':
          [
            {
              'name': 'Accuracy',
              'value': 0.64
            },
            {
              'name': 'False Positives',
              'value': 0.48
            }
          ]
        },
        {
          'task_name': 'Spam Detection',
          'metrics':
          [
            {
              'name': 'Accuracy',
              'value': 0.78
            },
            {
              'name': 'F1',
              'value': 0.71
            }
          ]
        }
      ]
    }
  }
  return general_data, models_data
