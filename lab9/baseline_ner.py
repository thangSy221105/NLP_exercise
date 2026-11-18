"""LAB 09: CoNLL-2003 NER baseline, BERT fine-tuning, and evaluation."""
from __future__ import annotations
import argparse, json, os, random, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"/"conll2003"
RESULT_DIR=ROOT/"result"
MODEL_DIR=RESULT_DIR/"model"
HF_HOME=ROOT/"data"/"huggingface"
os.environ.setdefault("HF_HOME",str(HF_HOME))
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

SEED=2026
MODEL_NAME="prajjwal1/bert-tiny"
MAX_LENGTH=128
BATCH_SIZE=32
EPOCHS=2
LABELS=["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"]
LABEL2ID={label:i for i,label in enumerate(LABELS)}
TYPE_NAMES=["PER","ORG","LOC","MISC"]
TYPE_TO_ID={"PER":1,"ORG":3,"LOC":5,"MISC":7}
ID_TO_TYPE={1:"PER",2:"PER",3:"ORG",4:"ORG",5:"LOC",6:"LOC",7:"MISC",8:"MISC"}
DATA_FILES={name:f"{name}-00000-of-00001.parquet" for name in ["train","validation","test"]}

def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def ensure_data():
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    base="https://huggingface.co/datasets/BramVanroy/conll2003/resolve/main/data/"
    for split,filename in DATA_FILES.items():
        path=DATA_DIR/filename
        if not path.exists():
            urllib.request.urlretrieve(base+filename+"?download=true",path)
    return DATA_DIR

def load_splits():
    ensure_data()
    result={}
    for split,filename in DATA_FILES.items():
        frame=pd.read_parquet(DATA_DIR/filename)
        result[split]=[
            {"id":str(row.id),"words":[str(w) for w in row.tokens],
             "labels":[int(x) for x in row.ner_tags],
             "pos":[int(x) for x in row.pos_tags]}
            for row in frame.itertuples(index=False)
        ]
    return result

def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True)

def id_to_tags(ids):
    return [LABELS[int(i)] for i in ids]

def bio_spans(tags):
    """Return (start, end-exclusive, entity type) from IOB2 tags."""
    spans=[]; start=None; kind=None
    def close(end):
        if start is not None and kind is not None: spans.append((start,end,kind))
    for i,tag in enumerate(tags):
        if tag.startswith("B-"):
            close(i); start=i; kind=tag[2:]
        elif tag.startswith("I-"):
            next_kind=tag[2:]
            if start is None or kind!=next_kind:
                close(i); start=i; kind=next_kind
        else:
            close(i); start=None; kind=None
    close(len(tags))
    return spans

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    f=2*p*r/(p+r) if p+r else 0.0
    return p,r,f

def evaluate_sequences(gold_sequences,pred_sequences):
    gold_all=set(); pred_all=set()
    token_gold=[]; token_pred=[]
    for si,(gold,pred) in enumerate(zip(gold_sequences,pred_sequences)):
        if len(gold)!=len(pred):
            raise ValueError("Gold and predicted word-level sequences must align.")
        token_gold.extend(gold); token_pred.extend(pred)
        gold_all.update((si,s,e,t) for s,e,t in bio_spans(id_to_tags(gold)))
        pred_all.update((si,s,e,t) for s,e,t in bio_spans(id_to_tags(pred)))
    common=gold_all & pred_all
    overall=prf(len(common),len(pred_all-common),len(gold_all-common))
    per_type={}
    for kind in TYPE_NAMES:
        g={x for x in gold_all if x[3]==kind}
        p={x for x in pred_all if x[3]==kind}
        tp=len(g&p); per_type[kind]=(*prf(tp,len(p-g),len(g-p)),len(g))
    overall=(*overall,len(gold_all),accuracy_score(token_gold,token_pred) if token_gold else 0.0)
    return overall,per_type,gold_all,pred_all

def length_bucket(n):
    return "4+" if n>=4 else str(n)

def entity_length_rows(model_name,gold_sequences,pred_sequences):
    counts={b:{"tp":0,"fp":0,"fn":0} for b in ["1","2","3","4+"]}
    for gold,pred in zip(gold_sequences,pred_sequences):
        gs=set(bio_spans(id_to_tags(gold))); ps=set(bio_spans(id_to_tags(pred)))
        for bucket in counts:
            g={x for x in gs if length_bucket(x[1]-x[0])==bucket}
            p={x for x in ps if length_bucket(x[1]-x[0])==bucket}
            counts[bucket]["tp"]+=len(g&p); counts[bucket]["fp"]+=len(p-g); counts[bucket]["fn"]+=len(g-p)
    rows=[]
    for bucket,c in counts.items():
        p,r,f=prf(c["tp"],c["fp"],c["fn"])
        rows.append({"model":model_name,"entity_length":bucket,"precision":p,"recall":r,"f1":f,
                     "gold_support":c["tp"]+c["fn"],"predicted_support":c["tp"]+c["fp"]})
    return rows

def word_shape(word):
    chars=[]
    for ch in word:
        cat="X" if ch.isupper() else "x" if ch.islower() else "d" if ch.isdigit() else ch
        if not chars or chars[-1]!=cat: chars.append(cat)
    return "".join(chars)[:12]

def token_features(words,pos,i):
    w=words[i]; low=w.lower()
    feat={"bias":1.0,"word.lower":low,"word[-3:]":low[-3:],"word[-2:]":low[-2:],
          "word[:2]":low[:2],"word[:3]":low[:3],"shape":word_shape(w),
          "isupper":w.isupper(),"istitle":w.istitle(),"isdigit":w.isdigit(),
          "has_digit":any(c.isdigit() for c in w),"length":min(len(w),20),
          "pos":str(pos[i]) if i<len(pos) else ""}
    for offset in [-2,-1,1,2]:
        j=i+offset
        prefix=f"{offset:+d}:"
        if 0<=j<len(words):
            feat[prefix+"word.lower"]=words[j].lower()
            feat[prefix+"shape"]=word_shape(words[j])
            feat[prefix+"pos"]=str(pos[j]) if j<len(pos) else ""
        else: feat[prefix+"BOS/EOS"]=True
    return feat

def sentence_features(record):
    return [token_features(record["words"],record.get("pos",[]),i)
            for i in range(len(record["words"]))]

def flatten_features(records):
    return [f for rec in records for f in sentence_features(rec)]

def flatten_labels(records):
    return [LABELS[int(tag)] for rec in records for tag in rec["labels"]]

def baseline_predict(train, test):
    start=time.perf_counter()
    vectorizer=DictVectorizer(sparse=True)
    x_train=vectorizer.fit_transform(flatten_features(train))
    y_train=flatten_labels(train)
    model=LogisticRegression(max_iter=300,tol=1e-3,class_weight="balanced",solver="lbfgs")
    model.fit(x_train,y_train)
    fit_seconds=time.perf_counter()-start
    x_test=vectorizer.transform(flatten_features(test))
    flat_pred=model.predict(x_test).tolist()
    preds=[]; cursor=0
    for rec in test:
        n=len(rec["words"]); preds.append([LABEL2ID[x] for x in flat_pred[cursor:cursor+n]]); cursor+=n
    params=int(model.coef_.size+model.intercept_.size)
    return preds,fit_seconds,params

def align_labels(word_ids,word_labels):
    aligned=[]; previous=None
    for wid in word_ids:
        if wid is None: aligned.append(-100)
        elif wid!=previous: aligned.append(int(word_labels[wid]))
        else: aligned.append(-100)
        previous=wid
    return aligned

def tokenize_and_align(tokenizer,words,word_labels,max_length=MAX_LENGTH):
    encoded=tokenizer(words,is_split_into_words=True,truncation=True,
                      max_length=max_length,add_special_tokens=True)
    word_ids=encoded.word_ids()
    return encoded,word_ids,align_labels(word_ids,word_labels)

class NERDataset(Dataset):
    def __init__(self,records,tokenizer,max_length=MAX_LENGTH):
        self.items=[]
        for rec in records:
            enc,wids,aligned=tokenize_and_align(tokenizer,rec["words"],rec["labels"],max_length)
            self.items.append({"input_ids":enc["input_ids"],"attention_mask":enc["attention_mask"],
             "labels":aligned,"word_ids":wids,"words":rec["words"],"gold":rec["labels"]})
    def __len__(self): return len(self.items)
    def __getitem__(self,index): return self.items[index]

def collate_ner(items,pad_id):
    length=max(len(x["input_ids"]) for x in items)
    input_ids=[]; masks=[]; labels=[]; word_ids=[]
    for x in items:
        pad=length-len(x["input_ids"])
        input_ids.append(x["input_ids"]+[pad_id]*pad)
        masks.append(x["attention_mask"]+[0]*pad)
        labels.append(x["labels"]+[-100]*pad)
        word_ids.append([(-1 if w is None else int(w)) for w in x["word_ids"]]+[-1]*pad)
    return {"input_ids":torch.tensor(input_ids,dtype=torch.long),
      "attention_mask":torch.tensor(masks,dtype=torch.long),
      "labels":torch.tensor(labels,dtype=torch.long),
      "word_ids":word_ids,
      "words":[x["words"] for x in items],"gold":[x["gold"] for x in items]}

def make_loader(records,tokenizer,shuffle=False,batch_size=BATCH_SIZE):
    ds=NERDataset(records,tokenizer)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,
       generator=torch.Generator().manual_seed(SEED),collate_fn=lambda x:collate_ner(x,tokenizer.pad_token_id))

def predictions_from_logits(logits,batch):
    token_pred=logits.argmax(-1).detach().cpu().tolist()
    gold_sequences=[]; pred_sequences=[]
    for i,(words,gold,wids) in enumerate(zip(batch["words"],batch["gold"],batch["word_ids"])):
        pred=[0]*len(words); seen=set()
        for j,wid in enumerate(wids):
            if wid>=0 and wid not in seen:
                pred[wid]=int(token_pred[i][j]); seen.add(wid)
        gold_sequences.append([int(x) for x in gold]); pred_sequences.append(pred)
    return gold_sequences,pred_sequences

def evaluate_model(model,loader,device):
    model.eval(); golds=[]; preds=[]; loss_sum=0.0; batches=0
    with torch.no_grad():
        for batch in loader:
            inputs={k:batch[k].to(device) for k in ["input_ids","attention_mask","labels"]}
            out=model(**inputs)
            loss_sum+=float(out.loss.item()); batches+=1
            g,p=predictions_from_logits(out.logits,batch)
            golds.extend(g); preds.extend(p)
    metrics,types,_,_=evaluate_sequences(golds,preds)
    return metrics,types,golds,preds,(loss_sum/max(batches,1))

def train_transformer(train,valid,test,tokenizer):
    seed_everything()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2label={i:l for i,l in enumerate(LABELS)}
    model=AutoModelForTokenClassification.from_pretrained(
      MODEL_NAME,num_labels=len(LABELS),id2label=id2label,label2id=LABEL2ID).to(device)
    tr=make_loader(train,tokenizer,True); va=make_loader(valid,tokenizer,False); te=make_loader(test,tokenizer,False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.01)
    best_state=None; best_f1=-1; history=[]
    start=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train(); total_loss=0.0; n=0
        for batch in tr:
            inputs={k:batch[k].to(device) for k in ["input_ids","attention_mask","labels"]}
            optimizer.zero_grad(set_to_none=True)
            out=model(**inputs); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); total_loss+=float(out.loss.item()); n+=1
        vm,_,_,_,vloss=evaluate_model(model,va,device)
        history.append({"epoch":epoch,"train_loss":total_loss/max(n,1),
                        "validation_loss":vloss,"validation_entity_f1":vm[2]})
        print(f"  epoch {epoch}/{EPOCHS}: train_loss={history[-1]['train_loss']:.4f}, val_entity_f1={vm[2]:.4f}",flush=True)
        if vm[2]>best_f1:
            best_f1=vm[2]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    training_seconds=time.perf_counter()-start
    model.load_state_dict(best_state)
    tm,ttypes,golds,preds,_=evaluate_model(model,te,device)
    params=sum(p.numel() for p in model.parameters())
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(MODEL_DIR); tokenizer.save_pretrained(MODEL_DIR)
    return model,tm,ttypes,golds,preds,training_seconds,params,history

def save_token_predictions(test,golds,preds):
    rows=[]
    for i,(rec,g,p) in enumerate(zip(test,golds,preds)):
        rows.append({"sentence_id":rec["id"],"text":" ".join(rec["words"]),
          "words_json":json.dumps(rec["words"],ensure_ascii=False),
          "gold_tags_json":json.dumps(id_to_tags(g)),
          "predicted_tags_json":json.dumps(id_to_tags(p))})
    pd.DataFrame(rows).to_csv(RESULT_DIR/"token_predictions.csv",index=False,encoding="utf-8-sig")

def tokenization_sample(test):
    rows=[r for r in test if len(r["words"])>1]
    lengths=np.array([len(r["words"]) for r in rows])
    median=float(np.median(lengths))
    groups={"short/entity":[],"short/non-entity":[],"long/entity":[],"long/non-entity":[]}
    for rec in rows:
        lg="short" if len(rec["words"])<=median else "long"
        ent=any(int(x)!=0 for x in rec["labels"])
        key=f"{lg}/{'entity' if ent else 'non-entity'}"
        if len(groups[key])<5: groups[key].append(rec)
    selected=[]
    for group,items in groups.items():
        selected.extend((group,x) for x in items)
    seen={r["id"] for _,r in selected}
    if len(selected)<20:
        for rec in rows:
            if rec["id"] not in seen:
                group=("short" if len(rec["words"])<=median else "long")+("/entity" if any(rec["labels"]) else "/non-entity")
                selected.append((group,rec)); seen.add(rec["id"])
                if len(selected)>=20: break
    return selected[:20]

def run_tokenization_analysis(tokenizer=None):
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    data=load_splits(); tokenizer=tokenizer or get_tokenizer()
    selected=tokenization_sample(data["test"]); rows=[]
    for group,rec in selected:
        enc=tokenizer(rec["words"],is_split_into_words=True,add_special_tokens=True,truncation=False)
        wids=enc.word_ids(); counts=Counter(w for w in wids if w is not None)
        subwords=sum(counts.values()); split_words=sum(n>1 for n in counts.values())
        rows.append({"sentence_id":rec["id"],"group":group,"words":len(rec["words"]),
          "tokens_ex_special":subwords,"subword_splits":split_words,
          "ratio_subwords_words":subwords/max(len(rec["words"]),1),
          "has_entity":any(rec["labels"]),
          "sentence":" ".join(rec["words"])})
    df=pd.DataFrame(rows); df.to_csv(RESULT_DIR/"tokenization_analysis.csv",index=False,encoding="utf-8-sig")
    summaries=[]
    df["length_group"]=df.words.map(lambda n:"short" if n<=np.median(df.words) else "long")
    for name,subset in [("short",df[df.length_group=="short"]),("long",df[df.length_group=="long"]),
       ("entity",df[df.has_entity]),("non-entity",df[~df.has_entity])]:
        summaries.append({"group":name,"sentences":len(subset),"mean_words":subset.words.mean(),
          "mean_subword_tokens":subset.tokens_ex_special.mean(),"mean_word_splits":subset.subword_splits.mean(),
          "mean_subwords_per_word":subset.ratio_subwords_words.mean()})
    summary=pd.DataFrame(summaries)
    summary.to_csv(RESULT_DIR/"tokenization_summary.csv",index=False,encoding="utf-8-sig")
    return df,summary

def find_context_examples(test,limit=3):
    occurrences=defaultdict(list)
    for rec in test:
        tags=id_to_tags(rec["labels"]); words=rec["words"]
        for i,(word,tag) in enumerate(zip(words,tags)):
            if tag!="O":
                kind=tag[2:]; key=word.casefold()
                context=" ".join(words[max(0,i-5):min(len(words),i+6)])
                item=(kind,context,rec["id"])
                if item not in occurrences[key]: occurrences[key].append(item)
    output=[]
    for surface,items in occurrences.items():
        types={x[0] for x in items}
        if len(types)>=2:
            first={}
            for item in items:
                if item[0] not in first: first[item[0]]=item
            chosen=list(first.values())[:2]
            if len(chosen)>=2:
                output.append({"surface_form":surface,"type_a":chosen[0][0],"context_a":chosen[0][1],
                    "type_b":chosen[1][0],"context_b":chosen[1][1]})
            if len(output)>=limit: break
    return pd.DataFrame(output)

def plot_confusion(golds,preds):
    y=[x for seq in golds for x in seq]; p=[x for seq in preds for x in seq]
    cm=confusion_matrix(y,p,labels=list(range(len(LABELS))))
    fig,ax=plt.subplots(figsize=(9,7)); im=ax.imshow(cm,cmap="Blues")
    fig.colorbar(im,ax=ax,label="Word count")
    ax.set_xticks(range(len(LABELS)),LABELS,rotation=45,ha="right")
    ax.set_yticks(range(len(LABELS)),LABELS)
    ax.set_xlabel("Predicted BIO tag"); ax.set_ylabel("Gold BIO tag")
    ax.set_title("CoNLL-2003 test confusion matrix (word-level)")
    for (i,j),v in np.ndenumerate(cm):
        if v>cm.max()*0.18: ax.text(j,i,str(v),ha="center",va="center",fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT/"confusion_matrix.png",dpi=160); plt.close(fig)

def plot_entity_length(rows):
    df=pd.DataFrame(rows)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for model,part in df.groupby("model"):
        ax.plot(part.entity_length,part.f1,marker="o",label=model)
    ax.set(xlabel="Gold entity length (words)",ylabel="Exact entity F1",ylim=(0,1),
       title="NER performance by entity length")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ROOT/"entity_length_analysis.png",dpi=160); plt.close(fig)

def write_error_analysis(test,golds,preds,results):
    fp_rows=[]; fn_rows=[]; boundary_rows=[]
    for si,(rec,g,p) in enumerate(zip(test,golds,preds)):
        gs=set(bio_spans(id_to_tags(g))); ps=set(bio_spans(id_to_tags(p)))
        text=" ".join(rec["words"])
        def phrase(span): return " ".join(rec["words"][span[0]:span[1]])
        for span in sorted(ps-gs):
            fp_rows.append({"sentence_id":rec["id"],"text":text,"gold_entity":"; ".join(phrase(x) for x in sorted(gs)),
              "predicted_entity":phrase(span),"gold_type":";".join(x[2] for x in sorted(gs)),
              "predicted_type":span[2],"error_type":"False positive / spurious entity",
              "explanation":"Model dự đoán entity không khớp span và type gold; kiểm tra lexical cue, context và domain."})
        for span in sorted(gs-ps):
            fn_rows.append({"sentence_id":rec["id"],"text":text,"gold_entity":phrase(span),
              "predicted_entity":"; ".join(phrase(x) for x in sorted(ps)),
              "gold_type":span[2],"predicted_type":";".join(x[2] for x in sorted(ps)),
              "error_type":"False negative / missing or wrong entity",
              "explanation":"Entity gold bị bỏ sót hoặc dự đoán sai span/type; kiểm tra từ hiếm và ngữ cảnh."})
        for gold in sorted(gs-ps):
            for pred in sorted(ps-gs):
                overlap=max(gold[0],pred[0])<min(gold[1],pred[1])
                boundary_diff=(gold[0],gold[1])!=(pred[0],pred[1])
                if overlap and boundary_diff:
                    boundary_rows.append({"sentence_id":rec["id"],"text":text,
                      "gold_entity":phrase(gold),"predicted_entity":phrase(pred),
                      "gold_type":gold[2],"predicted_type":pred[2],"error_type":"Boundary error",
                      "explanation":"Gold và predicted span chồng lấn nhưng ranh giới khác; kiểm tra token đầu/cuối và context."})
    fp=pd.DataFrame(fp_rows).head(5); fn=pd.DataFrame(fn_rows).head(5); bd=pd.DataFrame(boundary_rows).head(5)
    err=pd.concat([fp,fn,bd],ignore_index=True)
    err.to_csv(RESULT_DIR/"error_examples.csv",index=False,encoding="utf-8-sig")
    lines=["# Error Analysis — LAB 09","",
      "Đánh giá trên CoNLL-2003 test bằng exact entity match: entity đúng khi cả boundary và loại entity đều khớp.",
      "Từng nhóm dưới đây gồm tối đa 5 ví dụ được lấy từ test; lỗi boundary là hai span chồng lấn nhưng không trùng ranh giới.","",
      "| Error type | Text | Gold entity | Predicted entity | Gold type | Predicted type | Explanation |",
      "|---|---|---|---|---|---|---|"]
    for _,row in err.iterrows():
        clean=lambda x:str(x).replace("|","/").replace("\n"," ")
        lines.append("| "+" | ".join(clean(row[c]) for c in ["error_type","text","gold_entity","predicted_entity","gold_type","predicted_type","explanation"])+" |")
    lines += ["","## Interpretation","",
      "False positives thường phản ánh token viết hoa hoặc tên riêng bị gán nhầm thành entity. False negatives có thể xuất hiện khi entity hiếm hoặc context không đủ rõ.",
      "Boundary errors cho thấy model đoán được khu vực chứa tên nhưng chọn sai token bắt đầu/kết thúc. Label alignment dùng chiến lược chỉ gán nhãn subword đầu; subword sau nhận IGNORE_INDEX=-100 nên không trực tiếp tạo loss.",
      "Các giả thuyết trên cần được đối chiếu với từng câu; chúng không chứng minh duy nhất một nguyên nhân gây lỗi."]
    (ROOT/"error_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def write_detailed_error_report():
    """Write a Vietnamese report with five measured examples per error group."""
    errors=pd.read_csv(RESULT_DIR/"error_examples.csv").fillna("")
    labels={"False positive / spurious entity":"False Positive — dự đoán thừa/sai entity",
      "False negative / missing or wrong entity":"False Negative — bỏ sót/sai entity",
      "Boundary error":"Boundary Error — sai ranh giới"}
    lines=["# Error Analysis — LAB 09","",
      "Phân tích dùng dự đoán BERT-Tiny trên test split CoNLL-2003. Exact entity match yêu cầu đúng toàn bộ boundary và loại entity. Một câu có thể xuất hiện ở nhiều nhóm vì lỗi tách span đồng thời tạo false positive và false negative."]
    for key,title in labels.items():
        part=errors[errors.error_type==key].head(5)
        lines += ["",f"## {title} ({len(part)} ví dụ)","",
          "| Câu | Gold entity | Predicted entity | Gold type | Predicted type |",
          "|---|---|---|---|---|"]
        for _,row in part.iterrows():
            clean=lambda value:str(value).replace("|","/").replace("\n"," ")
            lines.append("| "+" | ".join(clean(row[c]) for c in ["text","gold_entity","predicted_entity","gold_type","predicted_type"])+" |")
    lines += ["","## Nhận xét từ các ví dụ","",
      "1. **Sai ranh giới tên nhiều từ:** `Nadim Ladki` bị tách thành hai span PER; `United Arab Emirates` bị chia thành các từ riêng. Exact match tính span gold là false negative và các span con không khớp là false positive.",
      "2. **Nhầm loại entity:** `AL-AIN` được gán PER trong khi gold là LOC. Viết hoa không đủ để xác định loại; context địa lý cần được học tốt hơn.",
      "3. **Tên nhiều từ và MISC:** `Asian Cup` là một span MISC trong gold nhưng model phân đoạn/đánh nhãn sai. Lỗi này kết hợp nhầm loại với sai boundary.",
      "4. **Quan hệ với metric:** token accuracy có thể ít giảm hơn entity F1 khi phần lớn từ đúng nhưng span chưa khớp. Confusion matrix token-level bổ trợ nhưng không thay thế exact-span metric.",
      "5. **Giới hạn:** đây là ví dụ định tính trên test, không chứng minh nguyên nhân duy nhất. Muốn kết luận entity hiếm khó hơn cần nối lỗi với tần suất trong train. Alignment dùng -100 cho subword phụ giúp loss đúng ở word level nhưng không bảo đảm span/type chính xác.",
      "","## Tóm tắt","",
      "Trong các ví dụ, lỗi nổi bật là tách sai tên nhiều từ và nhầm loại một số tên riêng. Các nhóm entity dài có F1 thấp hơn ở BERT-Tiny trong cấu hình này. Kết quả phụ thuộc seed 2026, CPU, checkpoint nhỏ và hai epoch."]
    (ROOT/"error_analysis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def run_all():
    seed_everything(); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    print("Loading CoNLL-2003 train/validation/test...")
    data=load_splits(); train,valid,test=data["train"],data["validation"],data["test"]
    tokenizer=get_tokenizer()
    print("Running Experiment 1: tokenizer/subword statistics...")
    tok_rows,tok_summary=run_tokenization_analysis(tokenizer)
    print(tok_summary.to_string(index=False))
    print("Training token-feature Logistic Regression baseline...")
    baseline,btime,bparams=baseline_predict(train,test)
    bm,btypes,bg,bp=evaluate_sequences([r["labels"] for r in test],baseline)
    print(f"Baseline exact entity F1={bm[2]:.4f}")
    print("Fine-tuning pretrained BERT-Tiny token classifier...")
    model,tm,ttypes,golds,preds,ttime,tparams,history=train_transformer(train,valid,test,tokenizer)
    print(f"Transformer exact entity F1={tm[2]:.4f}")
    result_rows=[
      {"model":"Logistic Regression token features","precision":bm[0],"recall":bm[1],"entity_f1":bm[2],
       "entity_support":bm[3],"token_accuracy":bm[4],"training_seconds":btime,"parameters":bparams},
      {"model":"BERT-Tiny fine-tuned","precision":tm[0],"recall":tm[1],"entity_f1":tm[2],
       "entity_support":tm[3],"token_accuracy":tm[4],"training_seconds":ttime,"parameters":tparams}]
    pd.DataFrame(result_rows).to_csv(RESULT_DIR/"results.csv",index=False,encoding="utf-8-sig")
    em=[]
    for name,types,overall in [("Logistic Regression token features",btypes,bm),("BERT-Tiny fine-tuned",ttypes,tm)]:
        em.append({"model":name,"entity_type":"Overall","precision":overall[0],"recall":overall[1],"f1":overall[2],"support":overall[3]})
        for kind,(pr,re,f1,support) in types.items():
            em.append({"model":name,"entity_type":kind,"precision":pr,"recall":re,"f1":f1,"support":support})
    pd.DataFrame(em).to_csv(RESULT_DIR/"entity_metrics.csv",index=False,encoding="utf-8-sig")
    len_rows=entity_length_rows("Logistic Regression token features",[r["labels"] for r in test],baseline)
    len_rows+=entity_length_rows("BERT-Tiny fine-tuned",golds,preds)
    pd.DataFrame(len_rows).to_csv(RESULT_DIR/"entity_length_analysis.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(RESULT_DIR/"learning_curves.csv",index=False,encoding="utf-8-sig")
    save_token_predictions(test,golds,preds)
    plot_confusion(golds,preds); plot_entity_length(len_rows)
    ctx=find_context_examples(test,limit=5)
    ctx.to_csv(RESULT_DIR/"context_examples.csv",index=False,encoding="utf-8-sig")
    write_error_analysis(test,golds,preds,result_rows)
    write_detailed_error_report()
    meta={"seed":SEED,"dataset":"CoNLL-2003","dataset_source":"BramVanroy/conll2003 Parquet mirror",
      "splits":{k:len(v) for k,v in data.items()},"model":MODEL_NAME,"tokenizer":MODEL_NAME,
      "alignment_strategy":"label first subword; -100 for special and continuation subwords",
      "max_length":MAX_LENGTH,"batch_size":BATCH_SIZE,"epochs":EPOCHS,
      "optimizer":"AdamW","learning_rate":3e-5,"hardware":"CUDA" if torch.cuda.is_available() else "CPU"}
    (RESULT_DIR/"reproducibility.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    from information_extraction import run_ie
    ie_metrics=run_ie()
    print(pd.DataFrame(result_rows).to_string(index=False))
    print("Saved Lab 9 results under",RESULT_DIR)
    return result_rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-all",action="store_true")
    parser.add_argument("--tokenization-only",action="store_true")
    args=parser.parse_args()
    if args.tokenization_only:
        _,summary=run_tokenization_analysis()
        print(summary.to_string(index=False))
    elif args.run_all: run_all()
    else: parser.error("Use --run-all or --tokenization-only.")

if __name__=="__main__": main()

