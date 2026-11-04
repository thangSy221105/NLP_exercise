import random, time, copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from preprocessing import load_data, build_vocabulary, encode_batch, retention
from positional_encoding import PositionalEncoding
from transformer_block import TransformerBlock
SEED=2026
EPOCHS=3
BATCH=64
torch.set_num_threads(4)
def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
class Classifier(nn.Module):
    def __init__(self,vocab,heads=4,position=True,pooling='mean',lstm=False):
        super().__init__(); self.position=position; self.pooling=pooling; self.lstm=lstm
        self.embedding=nn.Embedding(vocab,128,padding_idx=0)
        self.pe=PositionalEncoding(128)
        self.block=nn.LSTM(128,128,batch_first=True) if lstm else TransformerBlock(128,heads,256)
        self.linear=nn.Linear(128,1)
    def forward(self,ids):
        mask=ids.ne(0); x=self.embedding(ids)
        if self.lstm:
            x,_=self.block(x); z=x[torch.arange(len(ids)),mask.sum(1)-1]
        else:
            if self.position: x=self.pe(x)
            x=self.block(x,~mask)
            z=x[:,0] if self.pooling=='first' else (x*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
        return self.linear(z).squeeze(-1)
def loader(frame,vocab,length,shuffle=False):
    ids,_=encode_batch(frame.tokens.tolist(),vocab,length)
    return DataLoader(TensorDataset(torch.tensor(ids),torch.tensor(frame.label.to_numpy(),dtype=torch.float32)),batch_size=BATCH,shuffle=shuffle,generator=torch.Generator().manual_seed(SEED))
def evaluate(model,dl):
    model.eval(); probs=[]; labels=[]; loss=0
    with torch.no_grad():
        for x,y in dl:
            logits=model(x); loss+=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum').item(); probs.extend(logits.sigmoid().tolist()); labels.extend(y.tolist())
    pr,re,f1,_=precision_recall_fscore_support(labels,np.array(probs)>=.5,average='binary',zero_division=0)
    return dict(loss=loss/len(labels),accuracy=accuracy_score(labels,np.array(probs)>=.5),precision=pr,recall=re,f1=f1),probs
