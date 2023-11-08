"""
This class defines a config (essentialy a dict...), which is a combination of hyperparameters (like temperature) and additional parameters, such as "prompt_type" that affects the prompt generation process)
"""

class Config:
  def __init__(self, parameters_dict=None) -> None:
    if parameters_dict is None:
      self._parameters = dict()
    else:
      self._parameters = parameters_dict

  def set_parameter(self, key, value):
    self._parameters[key] = value
  
  def get_parameter(self, key, default=None):
    return self._parameters.get(key, default)

  def to_dict(self) -> dict:
    return self._parameters
