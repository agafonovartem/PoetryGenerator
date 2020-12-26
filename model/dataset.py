class Dataset:
  def __init__(self, path):
    self.path = path
    self.text = self.read_text(path)
    self.tokens = sorted(set(self.text)) 
    self.num_tokens = len(self.tokens) 
    self.token_to_idx = self.build_token_to_idx(self.tokens)
    self.idx_to_token = self.build_idx_to_token(self.token_to_idx)

  def read_text(self, path):
    with open(path, 'r') as iofile:
      text = iofile.readlines()

    res = ''
    for line in text:
      res += line
    text = res.lower()
    return text

  def build_token_to_idx(self, tokens):
    token_to_idx = {}
    i = 0
    for symbol in tokens:
      token_to_idx[symbol] = i
      i += 1
    return token_to_idx

  def build_idx_to_token(self, token_to_idx):
    return {y:x for x,y in token_to_idx.items()}
