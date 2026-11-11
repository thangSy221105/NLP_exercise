"""LAB 08: Pretrained BERT-Tiny and from-scratch sentiment experiments."""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, random, re, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR, RESULT_DIR = ROOT / "data", ROOT / "result"
HF_HOME = DATA_DIR / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED, MODEL_NAME = 2026, "prajjwal1/bert-tiny"
WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
torch.set_num_threads(min(4, torch.get_num_threads()))

def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_imdb():
    path = DATA_DIR / "imdb_10k.jsonl.gz"
    audit = json.loads((DATA_DIR / "dataset_audit.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != audit["subset_sha256"]:
        raise ValueError("IMDb subset fingerprint does not match its audit file.")
    df = pd.read_json(path, lines=True)[["text", "label", "split"]]
    return df, audit

def load_amazon():
    rows = []
    with (DATA_DIR / "amazon_cells_labelled.txt").open(encoding="utf-8") as f:
        for line in f:
            text, label = line.rstrip("\n").rsplit("\t", 1)
            rows.append({"text": text, "label": int(label)})
    return pd.DataFrame(rows)

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(frame, tokenizer, length):
    batch = tokenizer(frame.text.astype(str).tolist(), padding="max_length", truncation=True,
                      max_length=length, return_tensors="pt")
    items = [batch["input_ids"], batch["attention_mask"]]
    if "token_type_ids" in batch: items.append(batch["token_type_ids"])
    items.append(torch.tensor(frame.label.to_numpy(dtype=np.int64)))
    return TensorDataset(*items)

def loader(data, shuffle, batch=32):
    return DataLoader(data, batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(SEED))

def metrics_for(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= .5).astype(int)
    pr, re, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision": pr, "recall": re,
            "f1": f1, "loss": float("nan")}

def evaluate(model, data, device):
    model.eval(); ys, ps, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for batch in data:
            *xs, y = [t.to(device) for t in batch]
            kw = {"input_ids": xs[0], "attention_mask": xs[1], "labels": y}
            if len(xs) > 2: kw["token_type_ids"] = xs[2]
            out = model(**kw); total_loss += out.loss.item() * len(y); n += len(y)
            ys.extend(y.cpu().tolist()); ps.extend(out.logits.softmax(-1)[:, 1].cpu().tolist())
    result = metrics_for(ys, ps); result["loss"] = total_loss / max(n, 1)
    return result, np.asarray(ys), np.asarray(ps)

def fit_bert(train, valid, test, tokenizer, length=128, epochs=1, mode="fine_tune"):
    seed_everything(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    if mode == "frozen":
        for p in model.base_model.parameters(): p.requires_grad = False
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=1e-3 if mode == "frozen" else 2e-5)
    train_dl = loader(encode(train, tokenizer, length), True)
    valid_dl = loader(encode(valid, tokenizer, length), False)
    test_dl = loader(encode(test, tokenizer, length), False)
    best, best_f1, hist = None, -1, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = 0.0; n = 0
        for batch in train_dl:
            *xs, y = [t.to(device) for t in batch]
            kw = {"input_ids": xs[0], "attention_mask": xs[1], "labels": y}
            if len(xs) > 2: kw["token_type_ids"] = xs[2]
            optim.zero_grad(set_to_none=True); loss = model(**kw).loss
            loss.backward(); optim.step(); loss_sum += loss.item() * len(y); n += len(y)
        vm, _, _ = evaluate(model, valid_dl, device)
        hist.append({"epoch": epoch, "train_loss": loss_sum/max(n,1),
                     "val_loss": vm["loss"], "val_f1": vm["f1"]})
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    seconds = time.perf_counter() - start
    model.load_state_dict(best)
    result, y, p = evaluate(model, test_dl, device)
    total = sum(x.numel() for x in model.parameters())
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    return model, result, y, p, seconds, total, trainable, hist

class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size, length):
        super().__init__(); d=128
        self.embedding=nn.Embedding(vocab_size,d,padding_idx=0)
        layer=nn.TransformerEncoderLayer(d_model=d,nhead=4,dim_feedforward=256,
                                         dropout=.1,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.head=nn.Linear(d,2)
        pos=torch.arange(length).unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2)*(-np.log(10000.0)/d))
        pe=torch.zeros(length,d); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self, ids, mask):
        mask=mask.bool(); x=self.embedding(ids)+self.pe[:,:ids.shape[1]]
        x=self.encoder(x,src_key_padding_mask=~mask)
        x=(x*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp(min=1)
        return self.head(x)

def words(text): return WORD_RE.findall(str(text).lower())
def scratch_vocab(texts):
    counts=Counter(w for text in texts for w in words(text))
    top=sorted(counts,key=lambda w:(-counts[w],w))[:9998]
    return {"<PAD>":0,"<UNK>":1,**{w:i+2 for i,w in enumerate(top)}}
def scratch_data(frame, vocab, length):
    ids=np.zeros((len(frame),length),np.int64); mask=np.zeros_like(ids)
    for i,text in enumerate(frame.text):
        seq=[vocab.get(w,1) for w in words(text)[:length]] or [1]
        ids[i,:len(seq)]=seq; mask[i,:len(seq)]=1
    return TensorDataset(torch.tensor(ids),torch.tensor(mask),
                         torch.tensor(frame.label.to_numpy(dtype=np.int64)))
def fit_scratch(train, valid, test, length=64, epochs=2):
    seed_everything(); vocab=scratch_vocab(train.text); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ScratchTransformer(len(vocab),length).to(device)
    tr=loader(scratch_data(train,vocab,length),True,64)
    va=loader(scratch_data(valid,vocab,length),False,64)
    te=loader(scratch_data(test,vocab,length),False,64)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3); best=None; bestf=-1; start=time.perf_counter()
    for _ in range(epochs):
        model.train()
        for ids,mask,y in tr:
            ids,mask,y=ids.to(device),mask.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(ids,mask),y)
            loss.backward(); opt.step()
        model.eval(); yy=[]; pp=[]
        with torch.no_grad():
            for ids,mask,y in va:
                z=model(ids.to(device),mask.to(device)).softmax(-1)[:,1].cpu().tolist()
                yy.extend(y.tolist()); pp.extend(z)
        if metrics_for(yy,pp)["f1"]>bestf:
            bestf=metrics_for(yy,pp)["f1"]
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    seconds=time.perf_counter()-start; model.load_state_dict(best)
    model.eval(); yy=[]; pp=[]
    with torch.no_grad():
        for ids,mask,y in te:
            z=model(ids.to(device),mask.to(device)).softmax(-1)[:,1].cpu().tolist()
            yy.extend(y.tolist()); pp.extend(z)
    result=metrics_for(yy,pp); params=sum(x.numel() for x in model.parameters())
    return result,np.asarray(yy),np.asarray(pp),seconds,params,len(vocab)

def record(exp,condition,n,length,epochs,m,seconds,params,trainable):
    return {"experiment":exp,"condition":condition,"training_examples":int(n),
      "max_length":length,"epochs":epochs,"accuracy":m["accuracy"],"precision":m["precision"],
      "recall":m["recall"],"f1":m["f1"],"test_loss":m["loss"],
      "training_seconds":seconds,"parameters":params,"trainable_parameters":trainable,
      "estimated_attention_score_MiB_per_layer": (32*2*length**2*4/(1024**2) if exp=="sequence_length" else np.nan)}

def predictions_table(y,p,frame):
    pred=(p>=.5).astype(int)
    out=pd.DataFrame({"text":frame.text.tolist(),"true_label":y,"predicted_label":pred,
      "positive_probability":p,"confidence":np.maximum(p,1-p)})
    out["confidence_bin"]=pd.cut(out.confidence,[0,.6,.75,1.000001],
        labels=["Low","Medium","High"],include_lowest=True).astype(str)
    return out

def save_figures(rows, y, p, test, amazon_out):
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    pred=predictions_table(y,p,test)
    pred.to_csv(RESULT_DIR/"test_predictions.csv",index=False,encoding="utf-8-sig")
    cm=confusion_matrix(y,(p>=.5).astype(int),labels=[0,1])
    fig,ax=plt.subplots(figsize=(5.5,4.5)); im=ax.imshow(cm,cmap="Blues"); fig.colorbar(im,ax=ax)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Negative","Positive"],
      yticklabels=["Negative","Positive"],xlabel="Predicted",ylabel="True",
      title="IMDb test confusion matrix - BERT-Tiny")
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    fig.tight_layout(); fig.savefig(ROOT/"confusion_matrix.png",dpi=160); plt.close(fig)
    bins=pred.groupby("confidence_bin",observed=False).agg(
      count=("true_label","size"),mean_confidence=("confidence","mean"),
      accuracy=("true_label",lambda s:(s.to_numpy()==pred.loc[s.index,"predicted_label"].to_numpy()).mean()))
    bins.to_csv(RESULT_DIR/"confidence_analysis.csv",encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(6.5,4.5)); x=np.arange(len(bins))
    ax.bar(x-.18,bins.accuracy.fillna(0),.36,label="Accuracy")
    ax.bar(x+.18,bins.mean_confidence.fillna(0),.36,label="Mean confidence")
    ax.set_xticks(x,[f"{k} (n={int(n)})" for k,n in zip(bins.index,bins["count"].fillna(0))])
    ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("Confidence vs correctness")
    ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout()
    fig.savefig(ROOT/"confidence_analysis.png",dpi=160); plt.close(fig)
    errors=pred[pred.true_label!=pred.predicted_label].copy()
    errors["error_type"]=np.where(errors.predicted_label==1,"False Positive","False Negative")
    chosen=pd.concat([errors[errors.error_type=="False Positive"].nlargest(3,"confidence"),
      errors[errors.error_type=="False Negative"].nlargest(3,"confidence"),
      errors.nlargest(2,"confidence").assign(error_type="High-confidence wrong")])
    chosen.drop_duplicates(["text","true_label","predicted_label"]).to_csv(
      RESULT_DIR/"error_examples.csv",index=False,encoding="utf-8-sig")
    ds=pd.DataFrame([r for r in rows if r["experiment"]=="data_efficiency"]).sort_values("training_examples")
    fig,ax=plt.subplots(figsize=(6.5,4.5))
    ax.plot(ds.training_examples/6000*100,ds.accuracy,marker="o",label="BERT-Tiny accuracy")
    ax.plot(ds.training_examples/6000*100,ds.f1,marker="s",label="BERT-Tiny F1")
    scratch=pd.DataFrame([r for r in rows if r["experiment"]=="data_efficiency_scratch"]).sort_values("training_examples")
    if not scratch.empty:
        ax.plot(scratch.training_examples/6000*100,scratch.accuracy,marker="^",linestyle="--",label="Scratch accuracy")
        ax.plot(scratch.training_examples/6000*100,scratch.f1,marker="v",linestyle="--",label="Scratch F1")
    ax.set(xlabel="Training subset (%)",ylabel="Test score",ylim=(0,1),title="Data efficiency")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT/"data_efficiency.png",dpi=160); plt.close(fig)
    amazon_out.to_csv(RESULT_DIR/"domain_shift_predictions.csv",index=False,encoding="utf-8-sig")
    return pred

def run_all():
    seed_everything(); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    imdb,audit=load_imdb()
    tr=imdb[imdb.split=="train"].reset_index(drop=True)
    va=imdb[imdb.split=="validation"].reset_index(drop=True)
    te=imdb[imdb.split=="test"].reset_index(drop=True)
    tokenizer=get_tokenizer(); rows=[]; curves=[]
    # Experiment 1: same IMDb partitions and 64-token limit.
    m,y,p,secs,params,vocab=fit_scratch(tr,va,te,64,2)
    rows.append(record("pretrained_vs_scratch","Transformer from scratch",len(tr),64,2,m,secs,params,params))
    model,m,y,p,secs,params,trainable,h=fit_bert(tr,va,te,tokenizer,64,2)
    rows.append(record("pretrained_vs_scratch","BERT-Tiny fine-tuned",len(tr),64,2,m,secs,params,trainable))
    curves += [{"run":"Experiment 1 fine-tuning",**x} for x in h]
    saved_y,saved_p=y.copy(),p.copy(); del model
    # Experiment 2: frozen encoder and complete fine-tuning.
    for mode,name in [("frozen","Frozen encoder"),("fine_tune","Fine-tuning")]:
        model,m,_,_,secs,params,trainable,h=fit_bert(tr,va,te,tokenizer,128,2,mode)
        rows.append(record("feature_extraction_vs_fine_tuning",name,len(tr),128,2,m,secs,params,trainable))
        curves += [{"run":"Experiment 2 "+name,**x} for x in h]; del model
    # Experiment 3: pretrained data curve and a small scratch 10%/100% check.
    for percent in [10,25,50,100]:
        n=len(tr)*percent//100
        sub=tr.groupby("label",group_keys=False).sample(n=n//2,random_state=SEED).sample(frac=1,random_state=SEED)
        model,m,_,_,secs,params,trainable,h=fit_bert(sub.reset_index(drop=True),va,te,tokenizer,128,1)
        rows.append(record("data_efficiency",f"{percent}%",len(sub),128,1,m,secs,params,trainable))
        curves += [{"run":f"Experiment 3 {percent}%",**x} for x in h]; del model
    for percent in [10,100]:
        n=len(tr)*percent//100
        sub=tr.groupby("label",group_keys=False).sample(n=n//2,random_state=SEED).sample(frac=1,random_state=SEED).reset_index(drop=True)
        sm,sy,sp,ssecs,sparams,svocab=fit_scratch(sub,va,te,128,1)
        rows.append(record("data_efficiency_scratch",f"Scratch {percent}%",len(sub),64,1,sm,ssecs,sparams,sparams))
    # Experiment 4: same full train split and one epoch; vary sequence length.
    for length in [64,128,256]:
        model,m,_,_,secs,params,trainable,h=fit_bert(tr,va,te,tokenizer,length,1)
        rows.append(record("sequence_length",str(length),len(tr),length,1,m,secs,params,trainable))
        curves += [{"run":f"Experiment 4 length {length}",**x} for x in h]; del model
    # Experiment 5: train on IMDb, test on the independent Amazon product-review domain.
    amazon=load_amazon()
    model,dm,dy,dp,dsecs,dparams,dtrain,h=fit_bert(tr,va,amazon,tokenizer,64,2)
    rows.append(record("domain_shift","Amazon product reviews (out-of-domain)",len(tr),64,2,dm,dsecs,dparams,dtrain))
    amazon_pred=predictions_table(dy,dp,amazon); del model
    test_pred=save_figures(rows,saved_y,saved_p,te,amazon_pred)
    pd.DataFrame(rows).to_csv(RESULT_DIR/"results.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(curves).to_csv(RESULT_DIR/"learning_curves.csv",index=False,encoding="utf-8-sig")
    meta={"seed":SEED,"model":MODEL_NAME,"tokenizer":MODEL_NAME,
      "dataset":"Audited IMDb 10k subset; UCI Amazon product-review sentences for domain shift",
      "train_split":len(tr),"validation_split":len(va),"test_split":len(te),
      "max_sequence_lengths":[64,128,256],"batch_size":32,"optimizer":"AdamW",
      "fine_tuning_learning_rate":2e-5,"frozen_head_learning_rate":1e-3,
      "epochs":"2 main comparisons; 1 data/length sweeps",
      "hardware":"CUDA" if torch.cuda.is_available() else "CPU",
      "amazon_test_examples":len(amazon),"audit":audit}
    (RESULT_DIR/"reproducibility.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved results under",RESULT_DIR)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-all",action="store_true")
    args=p.parse_args()
    if not args.run_all: p.error("Use --run-all to run the complete experiment set.")
    run_all()

if __name__=="__main__": main()
