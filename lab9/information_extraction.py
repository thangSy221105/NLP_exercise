"""Mini information-extraction task for LAB 09."""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from baseline_ner import bio_spans, prf

ROOT=Path(__file__).resolve().parent
RESULT_DIR=ROOT/"result"
TYPE_TO_FIELD={"PER":"PERSON","ORG":"ORGANIZATION","LOC":"LOCATION"}

def render_entity(words):
    text=" ".join(words)
    text=re.sub(r"\s+([.,!?;:%)\]])",r"\1",text)
    text=re.sub(r"([(\[])\s+",r"\1",text)
    return text.strip()

def entities_from_tags(words,tags,include_misc=False):
    output={"PERSON":[],"ORGANIZATION":[],"LOCATION":[]}
    if include_misc: output["MISC"]=[]
    for start,end,kind in bio_spans(tags):
        field=TYPE_TO_FIELD.get(kind,"MISC" if include_misc and kind=="MISC" else None)
        if field:
            value=render_entity(words[start:end])
            if value not in output[field]: output[field].append(value)
    return output

def span_set(tags,allowed=("PER","ORG","LOC")):
    return {(start,end,kind) for start,end,kind in bio_spans(tags) if kind in allowed}

def run_ie(sample_size=30):
    path=RESULT_DIR/"token_predictions.csv"
    if not path.exists():
        raise FileNotFoundError("Run baseline_ner.py --run-all before the IE task.")
    df=pd.read_csv(path).head(sample_size)
    rows=[]; gold_all=set(); pred_all=set()
    for i,row in enumerate(df.itertuples(index=False)):
        words=json.loads(row.words_json)
        gold_tags=json.loads(row.gold_tags_json)
        pred_tags=json.loads(row.predicted_tags_json)
        gold=entities_from_tags(words,gold_tags)
        extracted=entities_from_tags(words,pred_tags)
        gs=span_set(gold_tags); ps=span_set(pred_tags)
        gold_all.update((i,*span) for span in gs)
        pred_all.update((i,*span) for span in ps)
        rows.append({"sentence_id":row.sentence_id,"text":row.text,
          "gold_structured_json":json.dumps(gold,ensure_ascii=False),
          "extracted_structured_json":json.dumps(extracted,ensure_ascii=False),
          "gold_entity_count":len(gs),"predicted_entity_count":len(ps),
          "exact_matches":len(gs&ps)})
    tp=len(gold_all&pred_all); fp=len(pred_all-gold_all); fn=len(gold_all-pred_all)
    p,r,f1=prf(tp,fp,fn)
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULT_DIR/"information_extraction.csv",index=False,encoding="utf-8-sig")
    metrics=pd.DataFrame([{"sample_sentences":len(rows),"precision":p,"recall":r,"f1":f1,
                           "gold_entities":len(gold_all),"predicted_entities":len(pred_all),
                           "true_positive":tp,"false_positive":fp,"false_negative":fn}])
    metrics.to_csv(RESULT_DIR/"information_extraction_metrics.csv",index=False,encoding="utf-8-sig")
    print("Mini IE (first",len(rows),"CoNLL-2003 test sentences):",metrics.iloc[0].to_dict())
    return metrics.iloc[0].to_dict()

if __name__=="__main__":
    run_ie()
