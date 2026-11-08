from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data import read_metadata
from .model import encode, BASE_SEED

def read_manifest(path):
 with path.open(encoding='utf8') as f:
  for line in f: yield json.loads(line)

def select_rows(manifest, mode):
 rows=[r for r in read_manifest(manifest) if r['source'] in (['tatoeba','wili'] if mode=='short' else ['wili'])]
 if mode=='short':
  counts=defaultdict(int); chosen=[]
  for r in rows:
   cap=400 if r['source']=='tatoeba' else 300
   if counts[r['label']]<cap: chosen.append(r); counts[r['label']]+=1
  return chosen
 return rows

def main():
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['short','long'],required=True); p.add_argument('--manifest',type=Path,default=Path('data/mixture/manifest.jsonl')); p.add_argument('--wili-dir',type=Path,default=Path('data/raw/wili-2018')); p.add_argument('--output',type=Path); p.add_argument('--report',type=Path); a=p.parse_args(); a.output=a.output or Path(f'artifacts/lid235_{a.mode}.npz'); a.report=a.report or Path(f'reports/{a.mode}_training.json')
 labels,names,_=read_metadata(a.wili_dir); idx={x:i for i,x in enumerate(labels)}; rows=select_rows(a.manifest,a.mode); rows=[r for r in rows if r['label'] in idx and r['text'].strip()]
 if not rows: raise ValueError('No rows selected')
 n_values=(2,3,4,5) if a.mode=='short' else (1,2,3,4); dimensions=768; sparsity=4; channels_count=2*len(n_values)
 train=[]; val=[]
 for r in rows:
  key=hashlib.blake2b((r['label']+'\0'+r['text']).encode('utf8'),digest_size=2).digest(); (val if int.from_bytes(key,'big')%10==0 else train).append(r)
 sums=np.zeros((channels_count,len(labels),dimensions),np.float32); counts=np.zeros(len(labels),np.int32)
 for k,r in enumerate(train):
  sums[:,idx[r['label']]]+=encode(r['text'],dimensions=dimensions,sparsity=sparsity,n_values=n_values); counts[idx[r['label']]]+=1
 norms=np.linalg.norm(sums,axis=2,keepdims=True); prototypes=np.rint(np.clip((sums/np.maximum(norms,1e-12))*127,-127,127)).astype(np.int8); weights=np.full(channels_count,1/channels_count,np.float32); bias=np.zeros(len(labels),np.float32)
 correct=0; total=0
 for r in val:
  v=encode(r['text'],dimensions=dimensions,sparsity=sparsity,n_values=n_values); raw=np.asarray([(prototypes[c].astype(np.int32)@v[c].astype(np.int32))/(127.0*127.0) for c in range(channels_count)],np.float32); s=(raw-raw.mean(1,keepdims=True))/np.maximum(raw.std(1,keepdims=True),1e-6); pred=int(np.argmax(np.einsum('ck,c->k',s,weights)+bias)); correct+=pred==idx[r['label']]; total+=1
 report={'mode':a.mode,'rows':len(rows),'train_rows':len(train),'validation_rows':len(val),'labels':len(labels),'dimensions':dimensions,'sparsity':sparsity,'channels':channels_count,'n_values':n_values,'validation_accuracy':correct/max(total,1),'source_counts':{s:sum(r['source']==s for r in rows) for s in ('wili','tatoeba')}}
 a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,prototypes=prototypes,channel_weights=weights,class_bias=bias,labels=np.asarray(labels),label_names=np.asarray(names),dimensions=np.asarray(dimensions),sparsity=np.asarray(sparsity),n_values=np.asarray(n_values),seed=np.asarray(BASE_SEED)); a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2),encoding='utf8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

