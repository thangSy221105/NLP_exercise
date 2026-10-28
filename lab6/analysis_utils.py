"""Inference diagnostics for attention heatmaps; weights are not causal importance."""
import numpy as np
import pandas as pd
import torch
from preprocessing import tokenize,encode_batch
from attention_models import predict_sequence

def trace_attention(item,text,vocabulary):
    words=tokenize(text)[:item["max_length"]]
    if not words:
        words=["<UNK>"]
    ids,lengths=encode_batch([words],vocabulary,item["max_length"])
    model=item["model"]
    model.eval()
    with torch.no_grad():
        ids_t,lengths_t=torch.from_numpy(ids),torch.from_numpy(lengths)
        pooled,A,V,mask=model.attention_components(ids_t,lengths_t)
        logits=model.linear(pooled).squeeze()
        mass=(A*mask[:,:,None]).sum(1)/lengths_t[:,None]
        # At eval: mean_i c_i = sum_j beta_j v_j, so sum_j contribution_j + bias = logit.
        signed=(V@model.linear.weight[0])*mass
        probability=float(torch.sigmoid(logits))
    n=len(words)
    return {"tokens":words,"weights":A[0,:n,:n].numpy(),"mass":mass[0,:n].numpy(),
        "contributions":signed[0,:n].numpy(),"probability":probability,"logit":float(logits),
        "bias":float(model.linear.bias[0].detach()),
        "decomposition_error":abs(float(signed.sum())+float(model.linear.bias[0].detach())-float(logits))}

def token_table(trace):
    return pd.DataFrame({"position":np.arange(len(trace["tokens"])),"token":trace["tokens"],
                         "received_mass":trace["mass"],"signed_value_contribution":trace["contributions"]})

def select_examples(item,test):
    predicted=item["test_pred"];actual=test.label.to_numpy()
    chosen=[]
    for correct in [True,False]:
        candidates=np.flatnonzero((predicted==actual)==correct).tolist()
        selected=[]
        for label in [0,1]:
            group=[i for i in candidates if actual[i]==label]
            if group:selected.append(min(group,key=lambda i:(test.iloc[i].length,i)))
        while len(selected)<3:
            left=[i for i in candidates if i not in selected]
            if not left:break
            selected.append(min(left,key=lambda i:(abs(test.iloc[i].length-item["max_length"]-1),i)))
        chosen.extend(selected)
    return chosen

def occlude_token(item,text,vocabulary,position):
    words=tokenize(text)
    del words[position]
    return float(predict_sequence(item,[" ".join(words)],vocabulary)[0])
