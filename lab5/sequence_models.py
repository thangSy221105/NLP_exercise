"""Sequence classifiers and reproducible CPU training for LAB 05."""
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from preprocessing import SEED,tokenize,encode_batch

EMBED_DIM=64
HIDDEN_DIM=64
BATCH_SIZE=128
LEARNING_RATE=0.002
EPOCHS=8
DROPOUT=0.2

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)

def metrics(y,p):
    pred=(np.asarray(p)>=.5).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    macro=precision_recall_fscore_support(y,pred,average="macro",zero_division=0)[2]
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"accuracy":float(accuracy_score(y,pred)),"precision":float(precision),
        "recall":float(recall),"f1":float(f1),"macro_f1":float(macro),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

class SequenceClassifier(nn.Module):
    def __init__(self,vocab_size,architecture):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,EMBED_DIM,padding_idx=0)
        recurrent={"RNN":nn.RNN,"LSTM":nn.LSTM,"GRU":nn.GRU}[architecture]
        self.recurrent=recurrent(EMBED_DIM,HIDDEN_DIM,batch_first=True,num_layers=1)
        self.dropout=nn.Dropout(DROPOUT)
        self.linear=nn.Linear(HIDDEN_DIM,1)
    def forward(self,ids,lengths):
        embedded=self.embedding(ids)
        outputs,_=self.recurrent(embedded)
        # Unidirectional: trailing PAD cannot affect earlier outputs.
        # Gather each review's last REAL timestep; padded outputs are never classified.
        final=outputs[torch.arange(ids.size(0),device=ids.device),lengths.to(ids.device)-1]
        return self.linear(self.dropout(final)).squeeze(-1)

def make_loader(frame,vocabulary,max_length,shuffle=False):
    ids,lengths=encode_batch(frame.tokens.tolist(),vocabulary,max_length)
    dataset=TensorDataset(torch.from_numpy(ids),torch.from_numpy(lengths),
                          torch.tensor(frame.label.to_numpy(),dtype=torch.float32))
    generator=torch.Generator().manual_seed(SEED)
    return DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=shuffle,num_workers=0,
                      generator=generator)

def evaluate_loader(model,loader):
    model.eval()
    criterion=nn.BCEWithLogitsLoss(reduction="sum")
    total=0.;probabilities=[];labels=[]
    with torch.no_grad():
        for ids,lengths,y in loader:
            logits=model(ids,lengths)
            total+=float(criterion(logits,y))
            probabilities.extend(torch.sigmoid(logits).tolist())
            labels.extend(y.tolist())
    return total/len(labels),np.asarray(probabilities),np.asarray(labels,dtype=int)

def fit_sequence(architecture,max_length,train,validation,vocabulary,root):
    root=Path(root)
    set_seed()
    config={"seed":SEED,"architecture":architecture,"max_length":max_length,
        "embedding_dimension":EMBED_DIM,"hidden_dimension":HIDDEN_DIM,"batch_size":BATCH_SIZE,
        "learning_rate":LEARNING_RATE,"epochs":EPOCHS,"dropout":DROPOUT,"optimizer":"Adam",
        "clip_grad_norm":1.0,"padding":"last_real_timestep","torch":torch.__version__,
        "dataset_sha256":hashlib.sha256((root/"data"/"imdb_10k.jsonl.gz").read_bytes()).hexdigest(),
        "vocab_sha256":hashlib.sha256(json.dumps(vocabulary,sort_keys=True).encode()).hexdigest(),
        "implementation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    checkpoints=root/"data"/"checkpoints"
    checkpoints.mkdir(exist_ok=True)
    weight_file=checkpoints/f"{architecture}_{max_length}_{key}.pt"
    metadata_file=weight_file.with_suffix(".json")
    model=SequenceClassifier(len(vocabulary),architecture)
    parameters=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if weight_file.exists() and metadata_file.exists():
        meta=json.loads(metadata_file.read_text(encoding="utf-8"))
        model.load_state_dict(torch.load(weight_file,map_location="cpu",weights_only=True))
        print(f"Loaded completed {architecture} length={max_length}; best epoch={meta['best_epoch']}",flush=True)
        return {"name":f"{architecture} L={max_length}","architecture":architecture,
                "max_length":max_length,"model":model,"config":config,**meta}
    start=perf_counter()
    train_loader=make_loader(train,vocabulary,max_length,True)
    val_loader=make_loader(validation,vocabulary,max_length)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    criterion=nn.BCEWithLogitsLoss()
    best_loss=float("inf");best_epoch=0;best_weights=None;history=[]
    process=psutil.Process()
    sampled_peak=process.memory_info().rss/1024**2
    for epoch in range(1,EPOCHS+1):
        epoch_start=perf_counter()
        model.train();loss_sum=0.;grad_norms=[]
        for ids,lengths,y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits=model(ids,lengths)
            loss=criterion(logits,y)
            loss.backward()
            grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),1.0)))
            optimizer.step()
            loss_sum+=float(loss.detach())*len(y)
        train_seconds=perf_counter()-epoch_start
        val_start=perf_counter()
        val_loss,p,y_true=evaluate_loader(model,val_loader)
        val_seconds=perf_counter()-val_start
        sampled_peak=max(sampled_peak,process.memory_info().rss/1024**2)
        row={"model":f"{architecture} L={max_length}","architecture":architecture,
            "max_length":max_length,"epoch":epoch,"train_loss":loss_sum/len(train),
            "validation_loss":val_loss,"train_seconds":train_seconds,"validation_seconds":val_seconds,
            "mean_preclip_grad_norm":float(np.mean(grad_norms)),
            "max_preclip_grad_norm":float(np.max(grad_norms)),**metrics(y_true,p)}
        history.append(row)
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"{architecture} L={max_length} epoch {epoch}/{EPOCHS}: train={row['train_loss']:.4f} val={val_loss:.4f} F1={row['f1']:.4f} train_s={train_seconds:.1f}",flush=True)
    training_seconds=perf_counter()-start
    model.load_state_dict(best_weights)
    val_loss,p,y=evaluate_loader(model,val_loader)
    meta={"history":history,"training_seconds":training_seconds,
          "fit_seconds":sum(row["train_seconds"] for row in history),
          "validation_seconds":sum(row["validation_seconds"] for row in history),
          "best_epoch":best_epoch,"validation_loss":val_loss,"parameters":parameters,
          "trainable_parameters":trainable,"parameter_mib":sum(p.numel()*p.element_size() for p in model.parameters())/1024**2,
          "sampled_process_rss_mib":sampled_peak,"validation":metrics(y,p),
          "vocabulary_size":len(vocabulary),"epochs":EPOCHS}
    torch.save(best_weights,weight_file)
    metadata_file.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return {"name":f"{architecture} L={max_length}","architecture":architecture,
            "max_length":max_length,"model":model,"config":config,**meta}

def predict_sequence(item,texts,vocabulary):
    tokens=[tokenize(text) for text in texts]
    ids,lengths=encode_batch(tokens,vocabulary,item["max_length"])
    item["model"].eval()
    probabilities=[]
    with torch.no_grad():
        for start in range(0,len(ids),BATCH_SIZE):
            logits=item["model"](torch.from_numpy(ids[start:start+BATCH_SIZE]),
                                  torch.from_numpy(lengths[start:start+BATCH_SIZE]))
            probabilities.extend(torch.sigmoid(logits).tolist())
    return np.asarray(probabilities)

def fit_baseline(train,validation):
    vectorizer=TfidfVectorizer(tokenizer=tokenize,token_pattern=None,lowercase=False,
        min_df=2,max_features=10000,sublinear_tf=True,norm="l2",dtype=np.float64)
    start=perf_counter()
    x=vectorizer.fit_transform(train.text)
    clf=LogisticRegression(C=1,solver="liblinear",max_iter=1000,random_state=SEED)
    clf.fit(x,train.label)
    elapsed=perf_counter()-start
    val_prob=clf.predict_proba(vectorizer.transform(validation.text))[:,1]
    return {"name":"TF-IDF + LR","vectorizer":vectorizer,"clf":clf,"training_seconds":elapsed,
            "fit_seconds":elapsed,"parameters":int(clf.coef_.size+clf.intercept_.size),
            "vocabulary_size":len(vectorizer.vocabulary_),"epochs":None,"max_length":None,
            "validation":metrics(validation.label,val_prob)}

def predict_baseline(item,texts):
    return item["clf"].predict_proba(item["vectorizer"].transform(texts))[:,1]

def metric_row(item,split):
    return {"record_type":"evaluation","model":item["name"],"split":split,"seed":SEED,
        "architecture":item.get("architecture","LR"),"max_length":item["max_length"],
        "embedding_dimension":EMBED_DIM if "architecture" in item else None,
        "hidden_dimension":HIDDEN_DIM if "architecture" in item else None,
        "vocabulary_size":item["vocabulary_size"],"parameters":item["parameters"],
        "epochs":item["epochs"],"best_epoch":item.get("best_epoch"),
        "training_seconds":item["training_seconds"],"fit_seconds":item["fit_seconds"],
        "validation_loss":item.get("validation_loss"),
        "parameter_mib":item.get("parameter_mib"),
        "sampled_process_rss_mib":item.get("sampled_process_rss_mib"),
        "metric_average":"binary positive=1; macro_f1 also reported",**item[split]}
