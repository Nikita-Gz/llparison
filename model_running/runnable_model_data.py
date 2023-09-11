class RunnableModel:
  def __init__(self, _id, owner, name, source, context_size, hf_inferable, available, price,
               discount=0.0) -> None:
    self._id = _id
    self.owner = owner
    self.name = name
    self.source = source
    self.context_size = context_size
    self.hf_inferable = hf_inferable
    self.available = available
    self.price = price
    self.discount = discount
  
  def get_price_with_discount(self):
    # discount = 0 -> no discount
    # discount = 1 -> free
    return self.price * (1 - self.discount)
