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
