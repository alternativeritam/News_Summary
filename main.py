import re
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch.optim as optim
import torchvision.models as models

model_path = "saved_model.pt"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

# loading the pre trained model

model.load_state_dict(torch.load(model_path))

#sending it to device
model = model.to(device)

tokenizer.encode(" TL;DR ")

# text processing function

def text_process(text):

  text = text.replace("\n","")
  text = text.replace("''","")
  text = text.replace("``","")
  final_text = text.split("@highlight")
  data = final_text[0].split("- --")
  try:
    return data[1]
  except:
    return data[0]

# chosing top-k tokens for the summary

def topk(probs, n=9):
    
    probs = torch.softmax(probs, dim= -1)
    
    
    tokensProb, topIx = torch.topk(probs, k=n)
    
    
    tokensProb = tokensProb / torch.sum(tokensProb)

    
    tokensProb = tokensProb.cpu().detach().numpy()

    
    choice = np.random.choice(n, 1, p = tokensProb)
    tokenId = topIx[choice][0]

    return int(tokenId)


# predicting summary

def model_infer(model, tokenizer, review, max_length=100):
    # Preprocess the init token (task designator)
    review_encoded = tokenizer.encode(review)
    result = review_encoded
    initial_input = torch.tensor(review_encoded).unsqueeze(0).to(device)

    with torch.set_grad_enabled(False):
        
        output = model(initial_input)

        
        logits = output.logits[0,-1]

        # Make a top-k choice and append to the result
        result.append(topk(logits))

        # For max_length times:
        for _ in range(max_length):
            
            input = torch.tensor(result).unsqueeze(0).to(device)
            output = model(input)
            logits = output.logits[0,-1]
            res_id = topk(logits)

            
            if res_id == tokenizer.eos_token_id:
                return tokenizer.decode(result)
            else: 
                result.append(res_id)
    
    return tokenizer.decode(result)


def final_summary(text):

    text = text_process(text)

    summary = model_infer(model,tokenizer,text+"TL;DR ").split("TL;DR ")[1].strip()

    return summary

text = input()

summary = final_summary(text)

print(text)
