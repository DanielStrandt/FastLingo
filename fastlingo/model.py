from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .tokenizer import channels

BASE_SEED = 20261108
HASH_MASK = (1 << 64) - 1

def splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over='ignore'):
        z=values+np.uint64(0x9E3779B97F4A7C15); z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9); z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB); return z^(z>>np.uint64(31))

def selected_channels(text: str, n_values: tuple[int,...]):
    all_channels=channels(text,max(n_values)); result=[]
    for view in range(2):
        for n in n_values: result.append(all_channels[view*max(n_values)+(n-1)])
    return result

def encode(text: str, *, dimensions: int, sparsity: int, n_values: tuple[int,...], seed: int=BASE_SEED) -> np.ndarray:
    out=np.zeros((2*len(n_values),dimensions),dtype=np.int8)
    for c,hashes in enumerate(selected_channels(text,n_values)):
        if hashes.size==0: continue
        unique,counts=np.unique(hashes,return_counts=True); w=(1.0+np.log(counts.astype(np.float32)))
        bases=unique[:,None]^np.uint64(seed+c*0x9E3779B1); offsets=np.arange(sparsity,dtype=np.uint64)[None,:]*np.uint64(0xD6E8FEB86659FD93); codes=splitmix64(bases+offsets)
        idx=(codes & np.uint64(dimensions-1)).astype(np.int64); signs=np.where((codes>>np.uint64(63))==0,1.0,-1.0).astype(np.float32)
        vec=np.bincount(idx.ravel(),weights=(w[:,None]*signs).ravel(),minlength=dimensions).astype(np.float32); norm=float(np.linalg.norm(vec))
        if norm: vec/=norm
        out[c]=np.clip(np.rint(vec*127.0),-127,127).astype(np.int8)
    return out

class CompactIdentifier:
    def __init__(self,path: str|Path):
        z=np.load(path,allow_pickle=False); self.labels=z['labels'].tolist(); self.names=z['label_names'].tolist(); self.prototypes=z['prototypes'].astype(np.int8); self.weights=z['channel_weights'].astype(np.float32); self.bias=z['class_bias'].astype(np.float32); self.dimensions=int(z['dimensions']); self.sparsity=int(z['sparsity']); self.n_values=tuple(z['n_values'].tolist()); self.seed=int(z['seed']); self._raw=np.empty((len(self.weights),len(self.labels)),np.float32)
    def predict(self,text:str,top_k:int=3):
        v=encode(text,dimensions=self.dimensions,sparsity=self.sparsity,n_values=self.n_values,seed=self.seed)
        for c in range(len(self.weights)): self._raw[c]=(self.prototypes[c].astype(np.int32)@v[c].astype(np.int32)).astype(np.float32)/(127.0*127.0)
        centered=self._raw-self._raw.mean(axis=1,keepdims=True); scaled=centered/np.maximum(self._raw.std(axis=1,keepdims=True),1e-6); scores=np.einsum('ck,c->k',scaled,self.weights)+self.bias; ranked=np.argsort(scores)[::-1]; order=ranked[:max(1,top_k)]; margin=float(scores[ranked[0]]-scores[ranked[1]])
        return {'label':self.labels[int(order[0])],'name':self.names[int(order[0])],'margin':margin,'top_k':[{'label':self.labels[int(i)],'name':self.names[int(i)],'score':float(scores[i])} for i in order]}

