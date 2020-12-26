import torch, torch.nn as nn
import torch.nn.functional as F

class CharLSTMLoop_hidden(nn.Module):
    def __init__(self, num_tokens=num_tokens, emb_size=16, rnn_num_units=64):
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.lstm = nn.LSTM(emb_size, rnn_num_units, batch_first=True)
        self.hid_to_logits = nn.Linear(rnn_num_units, num_tokens)
        self.hidden_dim = rnn_num_units
        
    def forward(self, x, hidden):
        # assert isinstance(x.data, torch.LongTensor)
        h_seq, hidden = self.lstm(self.emb(x), hidden)
        next_logits = self.hid_to_logits(h_seq)
        next_logp = F.log_softmax(next_logits, dim=-1)
        return next_logp, hidden 
        

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, batch_size, self.hidden_dim).zero_(),
                      weight.new(1, batch_size, self.hidden_dim).zero_())
        return hidden