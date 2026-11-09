from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import fasttext
import pycountry
from .data import iter_split, read_metadata
from .model import CompactIdentifier

def timed(fn, iterations=200):
    for _ in range(20): fn()
    values=[]
    for _ in range(iterations):
        start=time.perf_counter(); fn(); values.append((time.perf_counter()-start)*1000)
    values.sort(); return {'p50_ms':statistics.median(values),'p95_ms':values[int(.95*len(values))-1],'p99_ms':values[int(.99*len(values))-1],'mean_ms':statistics.fmean(values),'throughput_per_second':1000/statistics.fmean(values)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--fasttext-model',type=Path,default=Path('fasttext_lid.176.ftz')); p.add_argument('--wili-dir',type=Path,default=Path('data/raw/wili-2018')); p.add_argument('--compact-model',type=Path,default=Path('artifacts/lid235_long.npz')); p.add_argument('--report',type=Path,default=Path('reports/fasttext_comparison.json')); a=p.parse_args()
    ft=fasttext.load_model(str(a.fasttext_model)); compact=CompactIdentifier(a.compact_model); raw_ft_labels={x.removeprefix('__label__') for x in ft.get_labels()}
    ft_to_wili={code: (pycountry.languages.get(alpha_2=code).alpha_3 if pycountry.languages.get(alpha_2=code) else code) for code in raw_ft_labels}
    ft_to_wili.update({'iw':'heb','in':'ind','ji':'yid','jw':'jav','mo':'ron','sh':'srp'})
    wili_to_ft={wili:ft for ft,wili in ft_to_wili.items()}; labels,_,_=read_metadata(a.wili_dir); common=set(labels)&set(wili_to_ft)
    total=ft_correct=compact_correct=0; examples=[]
    for _,text,label in iter_split(a.wili_dir,'test'):
        if label not in common: continue
        total+=1; examples.append(text)
        predicted_ft=ft.predict(text,k=1)[0][0].removeprefix('__label__'); ft_correct += ft_to_wili.get(predicted_ft, predicted_ft) == label
        compact_correct += compact.predict(text,top_k=1)['label'] == label
    sample='This is a representative language identification sentence used for edge inference.'
    report={'dataset':'WiLI-2018 official test split, restricted to the semantic ISO-639 label intersection','common_labels':len(common),'examples':total,'fasttext_label_mapping':ft_to_wili,'fasttext':{'model_bytes':a.fasttext_model.stat().st_size,'accuracy':ft_correct/max(1,total),'latency':timed(lambda:ft.predict(sample,k=1))},'fastlingo_long':{'model_bytes':a.compact_model.stat().st_size,'accuracy':compact_correct/max(1,total),'latency':timed(lambda:compact.predict(sample,top_k=1))}}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

