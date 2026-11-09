from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import pandas as pd
import fasttext, pycountry
from .model import CompactIdentifier

def ft_mapping(ft):
    raw={x.removeprefix('__label__') for x in ft.get_labels()}; out={}
    for code in raw:
        item=pycountry.languages.get(alpha_2=code)
        out[code]=item.alpha_3 if item else code
    out.update({'iw':'heb','in':'ind','ji':'yid','jw':'jav','mo':'ron','sh':'srp'})
    return out

def macro_f1(y_true,y_pred,labels):
    vals=[]
    for label in labels:
        tp=sum(a==label and b==label for a,b in zip(y_true,y_pred)); fp=sum(a!=label and b==label for a,b in zip(y_true,y_pred)); fn=sum(a==label and b!=label for a,b in zip(y_true,y_pred))
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; vals.append(2*p*r/(p+r) if p+r else 0)
    return sum(vals)/max(1,len(vals))

def evaluate(rows, model, ft, mapping, common, per_label=100):
    seen=defaultdict(int)
    y=[]; hp=[]; fp=[]
    for text,label in rows:
        if label not in common: continue
        if seen[label] >= per_label: continue
        seen[label] += 1
        y.append(label); hp.append(model.predict(text,top_k=1)['label']); fp.append(mapping.get(ft.predict(text,k=1)[0][0].removeprefix('__label__'), 'und'))
    return {'examples':len(y),'labels':len(set(y)),'fastlingo_accuracy':sum(a==b for a,b in zip(y,hp))/max(1,len(y)),'fastlingo_macro_f1':macro_f1(y,hp,sorted(set(y))),'fasttext_accuracy':sum(a==b for a,b in zip(y,fp))/max(1,len(y)),'fasttext_macro_f1':macro_f1(y,fp,sorted(set(y)))}

def flores_rows(root):
    for path in sorted((root/'devtest').glob('*.devtest')):
        label=path.name.split('_')[0]
        for line in path.read_text(encoding='utf8').splitlines():
            if line.strip(): yield line.strip(),label

def main():
    p=argparse.ArgumentParser(); p.add_argument('--flores-dir',type=Path,default=Path('data/raw/flores200_dataset')); p.add_argument('--udhr-parquet',type=Path,default=Path('data/raw/udhr_test.parquet')); p.add_argument('--fasttext-model',type=Path,default=Path('fasttext_lid.176.ftz')); p.add_argument('--short-model',type=Path,default=Path('artifacts/lid235_short.npz')); p.add_argument('--long-model',type=Path,default=Path('artifacts/lid235_long.npz')); p.add_argument('--min-long-chars',type=int,default=240); p.add_argument('--per-label',type=int,default=100); p.add_argument('--report',type=Path,default=Path('reports/external_comparison.json')); a=p.parse_args()
    ft=fasttext.load_model(str(a.fasttext_model)); mapping=ft_mapping(ft); short=CompactIdentifier(a.short_model); long=CompactIdentifier(a.long_model); common_short=set(short.labels)&set(mapping.values()); common_long=set(long.labels)&set(mapping.values())
    flores=list(flores_rows(a.flores_dir)); udhr=pd.read_parquet(a.udhr_parquet,engine='fastparquet'); udhr_rows=[(str(row['sentence']),str(row['iso639-3'])) for _,row in udhr.iterrows() if len(str(row['sentence']))>=a.min_long_chars]
    result={'sampling':{'per_label':a.per_label,'long_min_chars':a.min_long_chars},'datasets':{'flores200_devtest_short':{'source':'FLORES-200 devtest','model':'short','label_intersection':len(common_short),**evaluate(flores,short,ft,mapping,common_short,a.per_label)},'udhr_lid_test_long':{'source':'UDHR-LID test, texts >= min-long-chars','model':'long','min_long_chars':a.min_long_chars,'label_intersection':len(common_long),**evaluate(udhr_rows,long,ft,mapping,common_long,a.per_label)}}}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()

