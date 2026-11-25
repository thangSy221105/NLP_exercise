"""LAB 10 extractive QA utilities and experiment pipeline."""
from __future__ import annotations
import argparse, gc, json, math, os, random, re, string, time, urllib.request
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parent
DATA_DIR=ROOT/"data"/"squad"
RAW_DIR=DATA_DIR/"raw"
RESULT_DIR=ROOT/"result"
MODEL_DIR=RESULT_DIR/"model"
HF_HOME=ROOT/"data"/"huggingface"
os.environ.setdefault("HF_HOME",str(HF_HOME))
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

SEED=2026
MODEL_NAME="prajjwal1/bert-tiny"
TRAIN_SIZE=3000
VAL_SIZE=300
TEST_SIZE=500
EXP1_TRAIN_SIZE=600
EXP1_EPOCHS=1
EPOCHS=3
BATCH_SIZE=16
LEARNING_RATE=3e-5
MAX_LENGTH=384
STRIDE=128
MAX_ANSWER_TOKENS=30
STOPWORDS=set("a an the is are was were be been being do does did what which who whom whose where when why how many much much of in on at to from by for with and or but as it this that these those i you he she they we their its his her my your our into about over under after before than then".split())


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4,os.cpu_count() or 1))


def ensure_squad_data():
    DATA_DIR.mkdir(parents=True,exist_ok=True); RAW_DIR.mkdir(parents=True,exist_ok=True)
    sample_paths={name:DATA_DIR/f"{name}.json" for name in ["train","validation","test"]}
    if all(p.exists() for p in sample_paths.values()):
        expected={"train":TRAIN_SIZE,"validation":VAL_SIZE,"test":TEST_SIZE}
        if all(len(json.loads(sample_paths[k].read_text(encoding="utf-8")))==n for k,n in expected.items()): return
    source={"train":"train-v1.1.json","dev":"dev-v1.1.json"}
    urls={"train":"https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
          "dev":"https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"}
    parsed={}
    for split,filename in source.items():
        path=RAW_DIR/filename
        if not path.exists():
            print(f"Downloading SQuAD {split} split...")
            urllib.request.urlretrieve(urls[split],path)
        with path.open(encoding="utf-8") as f: data=json.load(f)
        rows=[]
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context=paragraph["context"]
                for qa in paragraph["qas"]:
                    answers=[{"text":a["text"],"answer_start":int(a["answer_start"])} for a in qa.get("answers",[])]
                    if answers:
                        rows.append({"id":str(qa["id"]),"title":article.get("title",""),"context":context,
                                     "question":qa["question"],"answers":answers})
        parsed[split]=rows
    rng=random.Random(SEED)
    train=parsed["train"][:]
    dev=parsed["dev"][:]
    rng.shuffle(train); rng.shuffle(dev)
    if len(train)<TRAIN_SIZE or len(dev)<VAL_SIZE+TEST_SIZE:
        raise ValueError("SQuAD raw dataset is smaller than the requested seeded sample.")
    selected={"train":train[:TRAIN_SIZE],"validation":dev[:VAL_SIZE],"test":dev[VAL_SIZE:VAL_SIZE+TEST_SIZE]}
    for name,rows in selected.items():
        sample_paths[name].write_text(json.dumps(rows,ensure_ascii=False),encoding="utf-8")
    print("Prepared seeded SQuAD samples:", {k:len(v) for k,v in selected.items()})


def load_data():
    ensure_squad_data()
    output={}
    for split in ["train","validation","test"]:
        output[split]=json.loads((DATA_DIR/f"{split}.json").read_text(encoding="utf-8"))
    return output


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True,cache_dir=str(HF_HOME))


def align_answer_to_tokens(offsets,sequence_ids,answer_start,answer_text):
    """Map a context character span to inclusive model token positions, or (0,0) outside a window."""
    answer_end=int(answer_start)+len(answer_text)
    ctx=[i for i,sid in enumerate(sequence_ids) if sid==1 and offsets[i] is not None and offsets[i][1]>offsets[i][0]]
    if not ctx: return 0,0
    if offsets[ctx[0]][0]>answer_start or offsets[ctx[-1]][1]<answer_end: return 0,0
    start=ctx[0]
    while start<=ctx[-1] and offsets[start][0]<=answer_start: start+=1
    start-=1
    end=ctx[-1]
    while end>=ctx[0] and offsets[end][1]>=answer_end: end-=1
    end+=1
    if start not in ctx or end not in ctx or start>end: return 0,0
    return start,end


def tokenize_examples(examples,tokenizer,max_length=MAX_LENGTH,stride=STRIDE,with_labels=True,batch_size=32):
    features=[]
    for base in range(0,len(examples),batch_size):
        part=examples[base:base+batch_size]
        questions=[x["question"] for x in part]
        q_lengths=[len(ids) for ids in tokenizer(questions,add_special_tokens=False)["input_ids"]]
        usable_stride=max(0,min(stride,max_length-max(q_lengths or [0])-4))
        enc=tokenizer(questions,[x["context"] for x in part],
          truncation="only_second",max_length=max_length,stride=usable_stride,
          return_overflowing_tokens=True,return_offsets_mapping=True,padding=False)
        mapping=enc["overflow_to_sample_mapping"]
        for fi in range(len(enc["input_ids"])):
            local=int(mapping[fi]); ex=part[local]; seq=enc.sequence_ids(fi)
            raw_offsets=enc["offset_mapping"][fi]
            context_mask=[sid==1 and int(o[1])>int(o[0]) for sid,o in zip(seq,raw_offsets)]
            offsets=[(int(o[0]),int(o[1])) if context_mask[j] else None for j,o in enumerate(raw_offsets)]
            feature={"input_ids":list(enc["input_ids"][fi]),"attention_mask":list(enc["attention_mask"][fi]),
              "offset_mapping":offsets,"context_mask":context_mask,"example_index":base+local,
              "feature_index":len(features),"qid":ex["id"]}
            if "token_type_ids" in enc: feature["token_type_ids"]=list(enc["token_type_ids"][fi])
            if with_labels:
                ans=ex["answers"][0]
                st,en=align_answer_to_tokens(offsets,seq,int(ans["answer_start"]),ans["text"])
                feature["start_positions"]=st; feature["end_positions"]=en
            features.append(feature)
    return features


class QADataset(Dataset):
    def __init__(self,features): self.features=features
    def __len__(self): return len(self.features)
    def __getitem__(self,i): return self.features[i]


def qa_collate(items,pad_id):
    length=max(len(x["input_ids"]) for x in items)
    out={"input_ids":[],"attention_mask":[],"context_mask":[],"offset_mapping":[],
         "example_index":[],"feature_index":[],"qids":[]}
    has_types=any("token_type_ids" in x for x in items)
    if has_types: out["token_type_ids"]=[]
    has_labels="start_positions" in items[0]
    if has_labels: out["start_positions"]=[]; out["end_positions"]=[]
    for x in items:
        pad=length-len(x["input_ids"])
        out["input_ids"].append(x["input_ids"]+[pad_id]*pad)
        out["attention_mask"].append(x["attention_mask"]+[0]*pad)
        out["context_mask"].append(x["context_mask"]+[False]*pad)
        out["offset_mapping"].append(x["offset_mapping"]+[None]*pad)
        out["example_index"].append(x["example_index"]); out["feature_index"].append(x["feature_index"])
        out["qids"].append(x["qid"])
        if has_types: out["token_type_ids"].append(x.get("token_type_ids",[0]*len(x["input_ids"]))+[0]*pad)
        if has_labels:
            out["start_positions"].append(x["start_positions"]); out["end_positions"].append(x["end_positions"])
    for key in ["input_ids","attention_mask","token_type_ids","start_positions","end_positions"]:
        if key in out: out[key]=torch.tensor(out[key],dtype=torch.long)
    return out


def make_loader(features,tokenizer,shuffle=False,batch_size=BATCH_SIZE):
    gen=torch.Generator().manual_seed(SEED)
    return DataLoader(QADataset(features),batch_size=batch_size,shuffle=shuffle,
      generator=gen if shuffle else None,collate_fn=lambda xs:qa_collate(xs,tokenizer.pad_token_id))


def normalize_answer(text):
    def remove_articles(s): return re.sub(r"\b(a|an|the)\b"," ",s)
    def remove_punc(s): return "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(remove_articles(remove_punc((text or "").lower())).split())


def qa_f1(prediction,gold):
    p=normalize_answer(prediction).split(); g=normalize_answer(gold).split()
    if not p or not g: return float(p==g)
    common=Counter(p)&Counter(g); overlap=sum(common.values())
    if overlap==0: return 0.0
    precision=overlap/len(p); recall=overlap/len(g)
    return 2*precision*recall/(precision+recall)


def score_prediction(prediction,answers):
    refs=[a["text"] for a in answers] or [""]
    em=max(float(normalize_answer(prediction)==normalize_answer(ref)) for ref in refs)
    f1=max(qa_f1(prediction,ref) for ref in refs)
    return em,f1
import copy
from collections import defaultdict

def model_inputs(batch,device):
    keys=["input_ids","attention_mask","token_type_ids"]
    return {k:batch[k].to(device) for k in keys if k in batch}


def _predict_feature_batch(model,batch,device):
    inputs=model_inputs(batch,device)
    outputs=model(**inputs)
    return outputs.start_logits.detach().cpu(),outputs.end_logits.detach().cpu()


def predict_answers(model,examples,tokenizer,max_length=MAX_LENGTH,stride=STRIDE,batch_size=BATCH_SIZE):
    features=tokenize_examples(examples,tokenizer,max_length,stride,with_labels=False)
    loader=make_loader(features,tokenizer,False,batch_size)
    device=next(model.parameters()).device
    best={i:(-1e30,"",0,0,0,0) for i in range(len(examples))}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            starts,ends=_predict_feature_batch(model,batch,device)
            for r,feature_idx in enumerate(batch["feature_index"]):
                feature=features[feature_idx]
                valid=[i for i,x in enumerate(feature["context_mask"]) if x and feature["offset_mapping"][i] is not None]
                if not valid: continue
                top_s=sorted(valid,key=lambda i:float(starts[r,i]),reverse=True)[:20]
                top_e=sorted(valid,key=lambda i:float(ends[r,i]),reverse=True)[:20]
                ex_idx=feature["example_index"]
                context=examples[ex_idx]["context"]
                for s in top_s:
                    for e in top_e:
                        if e<s or e-s+1>MAX_ANSWER_TOKENS: continue
                        off_s=feature["offset_mapping"][s]; off_e=feature["offset_mapping"][e]
                        if off_s is None or off_e is None: continue
                        char_s, char_e=off_s[0],off_e[1]
                        if char_e<=char_s: continue
                        score=float(starts[r,s]+ends[r,e])
                        if score>best[ex_idx][0]:
                            best[ex_idx]=(score,context[char_s:char_e],char_s,char_e,s,e)
    return {i:{"score":v[0],"text":v[1],"start_char":v[2],"end_char":v[3],
               "start_token":v[4],"end_token":v[5]} for i,v in best.items()}


def evaluate_loss(model,features,tokenizer,batch_size=BATCH_SIZE):
    loader=make_loader(features,tokenizer,False,batch_size); device=next(model.parameters()).device
    total=0.0; count=0; model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs=model_inputs(batch,device)
            outputs=model(**inputs,start_positions=batch["start_positions"].to(device),
                          end_positions=batch["end_positions"].to(device))
            n=len(batch["start_positions"]); total+=float(outputs.loss)*n; count+=n
    return total/max(count,1)


def _example_metrics(examples,predictions):
    rows=[]
    for i,ex in enumerate(examples):
        p=predictions[i]; em,f1=score_prediction(p["text"],ex["answers"])
        gold=ex["answers"][0]; gs=int(gold["answer_start"]); ge=gs+len(gold["text"])
        rows.append({"id":ex["id"],"title":ex.get("title",""),"question":ex["question"],
          "context":ex["context"],"gold_answer":gold["text"],"prediction":p["text"],
          "gold_start_char":gs,"gold_end_char":ge,"pred_start_char":p["start_char"],
          "pred_end_char":p["end_char"],"pred_start_token":p["start_token"],
          "pred_end_token":p["end_token"],"answer_length":len(gold["text"].split()),
          "context_length":len(ex["context"].split()),"em":em,"f1":f1,
          "start_correct":int(p["start_char"]==gs),"end_correct":int(p["end_char"]==ge),
          "span_exact":int(p["start_char"]==gs and p["end_char"]==ge)})
    return pd.DataFrame(rows)


def train_model(train,valid,tokenizer,max_length=MAX_LENGTH,stride=STRIDE,epochs=EPOCHS,
                save_model=False,batch_size=BATCH_SIZE,verbose=True):
    seed_everything(); start_time=time.time(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features=tokenize_examples(train,tokenizer,max_length,stride,True)
    val_features=tokenize_examples(valid,tokenizer,max_length,stride,True)
    train_loader=make_loader(train_features,tokenizer,True,batch_size)
    model=AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME,cache_dir=str(HF_HOME)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)
    history=[]; best_f1=-1; best_state=None
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for batch in train_loader:
            inputs=model_inputs(batch,device)
            outputs=model(**inputs,start_positions=batch["start_positions"].to(device),
                          end_positions=batch["end_positions"].to(device))
            outputs.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); losses.append(float(outputs.loss.detach()))
        val_loss=evaluate_loss(model,val_features,tokenizer,batch_size)
        preds=predict_answers(model,valid,tokenizer,max_length,stride,batch_size)
        metrics=_example_metrics(valid,preds)
        row={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,
             "validation_em":float(metrics.em.mean()),"validation_f1":float(metrics.f1.mean())}
        history.append(row)
        if verbose: print(f"Epoch {epoch}/{epochs}: train loss={row['train_loss']:.4f}, val loss={val_loss:.4f}, EM={row['validation_em']:.3f}, F1={row['validation_f1']:.3f}")
        if row["validation_f1"]>best_f1:
            best_f1=row["validation_f1"]; best_state=copy.deepcopy(model.state_dict())
    if best_state is not None: model.load_state_dict(best_state)
    elapsed=time.time()-start_time
    if save_model:
        MODEL_DIR.mkdir(parents=True,exist_ok=True); model.save_pretrained(MODEL_DIR); tokenizer.save_pretrained(MODEL_DIR)
    return model,pd.DataFrame(history),elapsed


def _tokens_with_offsets(text):
    return [(m.group(),m.start(),m.end()) for m in re.finditer(r"\w+(?:['’]\w+)?|[^\w\s]",text)]


def keyword_baseline(examples):
    outputs={}
    for idx,ex in enumerate(examples):
        context=ex["context"]; qterms={w for w in re.findall(r"[a-z0-9]+",ex["question"].lower()) if w not in STOPWORDS}
        sentences=[(m.group(),m.start(),m.end()) for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)",context) if m.group().strip()]
        if not sentences: sentences=[(context,0,len(context))]
        def sentence_score(item):
            sent=item[0].lower(); terms=set(re.findall(r"[a-z0-9]+",sent))
            return len(qterms&terms)/(len(qterms) or 1)
        chosen=max(sentences,key=sentence_score); sent,base,_=chosen
        terms=_tokens_with_offsets(sent); question=ex["question"].lower()
        if not terms:
            outputs[idx]={"score":0,"text":"","start_char":0,"end_char":0,"start_token":0,"end_token":0}; continue
        candidates=[]
        for a in range(len(terms)):
            for b in range(a,min(len(terms),a+7)):
                text=sent[terms[a][1]:terms[b][2]].strip()
                if not text: continue
                words=text.split(); caps=sum(w[:1].isupper() for w in words); numeric=bool(re.search(r"\d",text))
                qtype=2 if re.search(r"\b(who|where|which|when)\b",question) else 0
                type_bonus=(0.45*caps if qtype else 0.25*numeric)
                # Prefer compact, information-bearing spans; lexical matching chooses the sentence.
                score=type_bonus+0.16*min(len(words),3)-0.045*len(words)
                if text.lower() in STOPWORDS: score-=1
                candidates.append((score,base+terms[a][1],base+terms[b][2],text,a,b))
        score,st,en,text,a,b=max(candidates,key=lambda x:x[0])
        outputs[idx]={"score":score,"text":text,"start_char":st,"end_char":en,"start_token":a,"end_token":b}
    return outputs


def _write_plot_learning(history,path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(history.epoch,history.train_loss,marker="o",label="Train loss")
    ax.plot(history.epoch,history.validation_loss,marker="o",label="Validation loss")
    ax.set(xlabel="Epoch",ylabel="Cross-entropy loss",title="Đường cong huấn luyện QA")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def _write_plot_metrics(results,matching,path):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(results.model,results.em,color=["#8da0cb","#fc8d62"])
    axes[0].set(title="Exact Match",ylabel="Score",ylim=(0,1)); axes[0].tick_params(axis="x",rotation=15)
    axes[1].bar(results.model,results.f1,color=["#8da0cb","#fc8d62"])
    axes[1].set(title="Token F1",ylabel="Score",ylim=(0,1)); axes[1].tick_params(axis="x",rotation=15)
    fig.suptitle("So sánh keyword baseline và Transformer")
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def _write_plot_lengths(lengths,path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(lengths))
    ax.bar(x-.18,lengths.em,width=.36,label="EM",color="#66c2a5")
    ax.bar(x+.18,lengths.f1,width=.36,label="F1",color="#fc8d62")
    ax.set_xticks(x,lengths.answer_length_group); ax.set_ylim(0,1)
    ax.set(xlabel="Số từ trong đáp án",ylabel="Điểm",title="Hiệu quả theo độ dài đáp án")
    ax.legend(); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def run_all(force=False):
    seed_everything(); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    data=load_data(); tokenizer=get_tokenizer()
    train=data["train"]; valid=data["validation"]; test=data["test"]
    print("SQuAD samples:", {k:len(v) for k,v in data.items()})
    # Experiment 1: vary only the maximum input length.
    exp_rows=[]
    exp_valid=valid[:200]
    for max_len in [128,256,384]:
        print(f"\nExperiment 1 — max_length={max_len}")
        model,hist,elapsed=train_model(train[:EXP1_TRAIN_SIZE],exp_valid,tokenizer,max_len,64,
                                       EXP1_EPOCHS,False,verbose=True)
        pred=predict_answers(model,test,tokenizer,max_len,64)
        scored=_example_metrics(test,pred)
        exp_rows.append({"max_length":max_len,"stride":64,"train_examples":EXP1_TRAIN_SIZE,
          "epochs":EXP1_EPOCHS,"training_seconds":elapsed,"test_em":float(scored.em.mean()),
          "test_f1":float(scored.f1.mean())})
        del model; gc.collect()
    length_results=pd.DataFrame(exp_rows)
    length_results.to_csv(RESULT_DIR/"context_length_results.csv",index=False)
    print("\nContext length results:\n",length_results.round(3).to_string(index=False))

    # Main fine-tuning run used for the remaining comparisons.
    print("\nTraining the main QA model...")
    model,history,training_seconds=train_model(train,valid,tokenizer,MAX_LENGTH,STRIDE,EPOCHS,True)
    history.to_csv(RESULT_DIR/"learning_curves.csv",index=False)
    _write_plot_learning(history,ROOT/"learning_curves.png")
    transformer_pred=predict_answers(model,test,tokenizer,MAX_LENGTH,STRIDE)
    transformer_rows=_example_metrics(test,transformer_pred)
    baseline_pred=keyword_baseline(test)
    baseline_rows=_example_metrics(test,baseline_pred)
    base_em=float(baseline_rows.em.mean()); base_f1=float(baseline_rows.f1.mean())
    qa_em=float(transformer_rows.em.mean()); qa_f1_score=float(transformer_rows.f1.mean())
    comparison=pd.DataFrame([
      {"model":"Keyword baseline","test_examples":len(test),"em":base_em,"f1":base_f1,"training_seconds":0},
      {"model":"BERT-Tiny fine-tuned","test_examples":len(test),"em":qa_em,"f1":qa_f1_score,"training_seconds":training_seconds}])
    comparison.to_csv(RESULT_DIR/"results.csv",index=False)
    transformer_rows.to_csv(RESULT_DIR/"token_predictions.csv",index=False)

    # Required partial-match cases (EM=0 but token F1>0).
    partial=transformer_rows[(transformer_rows.em==0)&(transformer_rows.f1>0)].sort_values("f1",ascending=False).head(20)
    partial.to_csv(RESULT_DIR/"em0_f1positive_examples.csv",index=False)

    # Context/question lexical matching: compare low and high overlap quartiles.
    question_terms=[]; overlap_values=[]
    for _,row in transformer_rows.iterrows():
        q={w for w in re.findall(r"[a-z0-9]+",row.question.lower()) if w not in STOPWORDS}
        ctx=row.context.lower().replace(str(row.gold_answer).lower()," ")
        c=set(re.findall(r"[a-z0-9]+",ctx))
        overlap_values.append(len(q&c)/(len(q) or 1)); question_terms.append(len(q))
    transformer_rows["lexical_overlap"]=overlap_values
    q25=float(np.quantile(overlap_values,.25)); q75=float(np.quantile(overlap_values,.75))
    transformer_rows["matching_group"]=np.where(transformer_rows.lexical_overlap<=q25,"Low overlap",
      np.where(transformer_rows.lexical_overlap>=q75,"High overlap","Middle"))
    transformer_rows[["id","question","gold_answer","prediction","lexical_overlap","matching_group","em","f1"]].to_csv(RESULT_DIR/"question_context_matching.csv",index=False)
    matching=(transformer_rows[transformer_rows.matching_group!="Middle"].groupby("matching_group")
       .agg(count=("id","count"),mean_overlap=("lexical_overlap","mean"),em=("em","mean"),f1=("f1","mean"))
       .reset_index())
    matching.to_csv(RESULT_DIR/"question_context_matching_metrics.csv",index=False)

    # Answer-length buckets.
    bins=[0,1,3,6,1000]; labels=["1","2–3","4–6","7+"]
    transformer_rows["answer_length_group"]=pd.cut(transformer_rows.answer_length,bins=bins,labels=labels)
    lengths=(transformer_rows.groupby("answer_length_group",observed=False)
      .agg(count=("id","count"),em=("em","mean"),f1=("f1","mean"))
      .reset_index())
    lengths.to_csv(RESULT_DIR/"answer_length_analysis.csv",index=False)

    # Start/end/span error slices; a row may belong to more than one slice.
    wrong_start=transformer_rows[(transformer_rows.start_correct==0)&(transformer_rows.end_correct==1)].head(10)
    wrong_end=transformer_rows[(transformer_rows.start_correct==1)&(transformer_rows.end_correct==0)].head(10)
    wrong_span=transformer_rows[transformer_rows.span_exact==0].head(10)
    error_frames=[]
    for label,frame in [("Wrong start",wrong_start),("Wrong end",wrong_end),("Wrong span",wrong_span)]:
        part=frame.copy(); part["error_group"]=label; error_frames.append(part)
    errors=pd.concat(error_frames,ignore_index=True).drop_duplicates(subset=["id","error_group"])
    errors.to_csv(RESULT_DIR/"error_examples.csv",index=False)
    emd=["# Error analysis — Lab 10", "",
      f"BERT-Tiny fine-tuned dự đoán trên {len(test)} câu hỏi. Có {int((transformer_rows.start_correct==0).sum())} câu sai vị trí bắt đầu, {int((transformer_rows.end_correct==0).sum())} câu sai vị trí kết thúc và {int((transformer_rows.span_exact==0).sum())} câu có span chưa khớp chính xác.",
      "", "## Bảng ví dụ lỗi", "",
      "| Nhóm lỗi | Câu hỏi | Đáp án chuẩn | Dự đoán | Giải thích theo ranh giới | EM | F1 |", "|---|---|---|---|---|---:|---:|"]
    for _,r in errors.head(30).iterrows():
        q=str(r.question).replace("|","/"); g=str(r.gold_answer).replace("|","/"); p=str(r.prediction).replace("|","/")
        if r.error_group=="Wrong start":
            explanation="Sai start boundary; end chính xác."
        elif r.error_group=="Wrong end":
            explanation="Sai end boundary; start chính xác."
        else:
            explanation=("Partial answer/span boundary sai." if r.f1>0 else "Không giao token với gold; cần đọc context để xác định có phải distractor hay không.")
        emd.append(f"| {r.error_group} | {q} | {g} | {p} | {explanation} | {r.em:.0f} | {r.f1:.3f} |")
    emd += ["", "## Nhận xét", "",
      "Sai start xảy ra khi span bắt đầu quá sớm hoặc quá muộn; sai end xảy ra khi span bị cắt cụt hoặc lấy thừa từ. Các lỗi này có thể đồng thời xuất hiện trong cùng một dự đoán. EM yêu cầu toàn bộ span khớp sau chuẩn hóa, còn F1 vẫn ghi nhận các từ giao nhau nên một phần câu trả lời có thể đúng dù EM bằng 0.",
      "", "Các nguyên nhân cần kiểm tra gồm câu trả lời dài, nhiều thực thể gần nhau, ngữ cảnh dài có từ khóa gây nhiễu, câu hỏi được diễn đạt khác với đoạn văn, và lỗi ánh xạ từ character offset sang token. Bảng ví dụ ở trên là các quan sát từ test split; phần nguyên nhân là giả thuyết cần đối chiếu thủ công với từng context.",
      "", "## Số lượng theo nhóm", "",
      f"- Wrong start: {len(wrong_start)} ví dụ được lưu (tổng lỗi: {int((transformer_rows.start_correct==0).sum())}).",
      f"- Wrong end: {len(wrong_end)} ví dụ được lưu (tổng lỗi: {int((transformer_rows.end_correct==0).sum())}).",
      f"- Wrong span: {len(wrong_span)} ví dụ được lưu (tổng lỗi: {int((transformer_rows.span_exact==0).sum())})."]
    (ROOT/"error_analysis.md").write_text("\n".join(emd)+"\n",encoding="utf-8")

    _write_plot_metrics(comparison,matching,ROOT/"em_f1_analysis.png")
    _write_plot_lengths(lengths,ROOT/"answer_length_analysis.png")
    reproducibility={"dataset":"SQuAD v1.1","seed":SEED,"model":MODEL_NAME,
      "train_examples":len(train),"validation_examples":len(valid),"test_examples":len(test),
      "max_length":MAX_LENGTH,"stride":STRIDE,"batch_size":BATCH_SIZE,
      "learning_rate":LEARNING_RATE,"epochs":EPOCHS,"optimizer":"AdamW",
      "device":"CUDA" if torch.cuda.is_available() else "CPU","training_seconds":training_seconds}
    (RESULT_DIR/"reproducibility.json").write_text(json.dumps(reproducibility,indent=2),encoding="utf-8")
    print("\nModel comparison:\n",comparison.round(3).to_string(index=False))
    print("\nQuestion/context matching:\n",matching.round(3).to_string(index=False))
    print("\nAnswer length:\n",lengths.round(3).to_string(index=False))
    print(f"\nPartial matches saved: {len(partial)}; error examples: start={len(wrong_start)}, end={len(wrong_end)}, span={len(wrong_span)}")
    print("Saved CSV tables under",RESULT_DIR)
    del model; gc.collect()
    return {"context_length":length_results,"comparison":comparison,"history":history,
      "predictions":transformer_rows,"matching":matching,"answer_length":lengths,"errors":errors}


def main():
    parser=argparse.ArgumentParser(description="Run Lab 10 extractive QA experiments")
    parser.add_argument("--run-all",action="store_true",help="run all training and evaluation experiments")
    parser.add_argument("--prepare-data",action="store_true",help="prepare deterministic SQuAD sample files")
    args=parser.parse_args()
    if args.prepare_data: ensure_squad_data()
    if args.run_all or not args.prepare_data: run_all()

if __name__=="__main__": main()



