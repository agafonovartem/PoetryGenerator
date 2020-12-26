import string
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import random

#from model import CharLSTMLoop_hidden


def generate_text_hidden(length, model, dataset, initial = ' ', temperature=1.0):
    x_sequence = [dataset.token_to_idx[token] for token in initial]
    
    x_sequence = torch.tensor([x_sequence], dtype=torch.int64).unsqueeze(0)

    if (torch.cuda.is_available()):
      x_sequence = x_sequence.cuda() 
    
    
    h = model.init_hidden(1)

    #start generating
    for _ in range(length):

        #________________________________
        # x_sequence[ :, :, -1] IS the last symbol of the sequence
        #________________________________

        h = tuple([e.data for e in h])

        logp_next, h = model(x_sequence[ :, :, -1], h)
       
        if (torch.cuda.is_available()):
          p_next = F.softmax(logp_next / temperature, dim=-1).cpu().data.numpy()[0][-1]
        else:
          p_next = F.softmax(logp_next / temperature, dim=-1).data.numpy()[0][-1]
        
        
        # sample next token and push it back into x_sequence
        next_ix = np.random.choice(dataset.num_tokens, p = p_next)
        next_ix = torch.tensor([[next_ix]], dtype=torch.int64)
        if (torch.cuda.is_available()):
          next_ix = next_ix.cuda()
        x_sequence = torch.cat([x_sequence[:, -1], next_ix], dim=1)
        x_sequence = x_sequence.unsqueeze(0)

    
    if (torch.cuda.is_available()):
          return ''.join([dataset.tokens[ix] for ix in x_sequence.cpu().data.numpy()[0][0]])
    return ''.join([dataset.tokens[ix] for ix in x_sequence.data.numpy()[0][0]])

