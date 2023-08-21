class TaskOutput:
  def __init__(self, metrics: dict, model_outputs: list, interpreted_outputs: list, input_codes: list, errors) -> None:
    self.metrics = metrics
    self.model_outputs = model_outputs
    self.interpreted_outputs = interpreted_outputs
    self.input_codes = input_codes # input code will allow Task class to return the exact input
    self.errors = errors
