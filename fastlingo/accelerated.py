"""Compiled numeric inference path with the same model math as model.py."""
from __future__ import annotations
import numpy as np
from numba import njit, types
from numba.typed import Dict
from .tokenizer import channels

@njit(cache=True, fastmath=True)
def _ascii_channels(data, max_n):
    channel_count=2*max_n; capacity=2*len(data)+2; values=np.zeros((channel_count,capacity),dtype=np.uint64); lengths=np.zeros(channel_count,dtype=np.int64)
    for view in range(2):
        codes=np.empty(capacity,dtype=np.uint64); pos=0; codes[pos]=0x110000; pos+=1; codes[pos]=0x110002; pos+=1
        for byte in data:
            if byte==32:
                codes[pos]=0x110003; pos+=1; codes[pos]=0x110002; pos+=1
            else:
                value=byte
                if view==1 and value>=65 and value<=90: value+=32
                codes[pos]=value; pos+=1
        codes[pos]=0x110003; pos+=1; codes[pos]=0x110001; pos+=1
        for n in range(1,max_n+1):
            c=view*max_n+n-1; size=pos-n+1; salt=np.uint64(((view+1)*0x9E3779B185EBCA87 ^ n*0xC2B2AE3D27D4EB4F)&0xFFFFFFFFFFFFFFFF); mult=np.uint64(0x100000001B3)
            for start in range(size):
                h=np.uint64((0xCBF29CE484222325 ^ int(salt))&0xFFFFFFFFFFFFFFFF)
                for offset in range(n): h=h*mult+codes[start+offset]+np.uint64(1)
                h^=h>>np.uint64(30); h*=np.uint64(0xBF58476D1CE4E5B9); h^=h>>np.uint64(27); h*=np.uint64(0x94D049BB133111EB); h^=h>>np.uint64(31); values[c,start]=h
            lengths[c]=size
    return values,lengths

@njit(cache=True, fastmath=True)
def _encode_kernel(flat, offsets, dimensions, sparsity, seed, channel_count):
    out=np.zeros((channel_count,dimensions),dtype=np.int8)
    for c in range(channel_count):
        start=offsets[c]; end=offsets[c+1]
        counts=Dict.empty(key_type=types.uint64,value_type=types.int64)
        for i in range(start,end): counts[flat[i]]=counts.get(flat[i],0)+1
        vec=np.zeros(dimensions,dtype=np.float32)
        for key,count in counts.items():
            weight=1.0+np.log(float(count))
            base=np.uint64(key)^np.uint64(seed+c*0x9E3779B1)
            for j in range(sparsity):
                z=base+np.uint64(j)*np.uint64(0xD6E8FEB86659FD93)
                z=(z+np.uint64(0x9E3779B97F4A7C15)); z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9); z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB); z=z^(z>>np.uint64(31))
                idx=int(z & np.uint64(dimensions-1)); sign=1.0 if (z>>np.uint64(63))==0 else -1.0; vec[idx]+=weight*sign
        norm=np.sqrt(np.sum(vec*vec))
        if norm>0:
            for d in range(dimensions):
                q=vec[d]/norm*127.0
                if q>127.0: q=127.0
                elif q < -127.0: q=-127.0
                out[c,d]=np.int8(np.rint(q))
    return out

def encode_accelerated(text: str, *, dimensions: int, sparsity: int, n_values: tuple[int,...], seed: int):
    if all(ord(ch)<128 for ch in text):
        raw,lengths=_ascii_channels(np.frombuffer(text.encode('ascii'),dtype=np.uint8),max(n_values)); selected=[raw[view*max(n_values)+n-1,:lengths[view*max(n_values)+n-1]] for view in range(2) for n in n_values]
    else:
        selected=[]
        all_channels=channels(text,max(n_values))
        for view in range(2):
            for n in n_values: selected.append(all_channels[view*max(n_values)+(n-1)])
    offsets=np.zeros(len(selected)+1,dtype=np.int64); total=0
    for i,arr in enumerate(selected): total+=len(arr); offsets[i+1]=total
    flat=np.empty(total,dtype=np.uint64); pos=0
    for arr in selected: flat[pos:pos+len(arr)]=arr; pos+=len(arr)
    return _encode_kernel(flat,offsets,dimensions,sparsity,seed,len(selected))

class AcceleratedIdentifier:
    def __init__(self,path):
        from .model import CompactIdentifier
        base=CompactIdentifier(path); self.__dict__.update(base.__dict__); self._prototypes_i32=self.prototypes.astype(np.int32)
    def predict(self,text,top_k=3):
        v=encode_accelerated(text,dimensions=self.dimensions,sparsity=self.sparsity,n_values=self.n_values,seed=self.seed).astype(np.int32)
        raw=np.einsum('cld,cd->cl',self._prototypes_i32,v).astype(np.float32)/(127.0*127.0)
        centered=raw-raw.mean(axis=1,keepdims=True); scaled=centered/np.maximum(raw.std(axis=1,keepdims=True),1e-6); scores=scaled.T@self.weights+self.bias; ranked=np.argsort(scores)[::-1]; order=ranked[:max(1,top_k)]
        return {'label':self.labels[int(order[0])],'name':self.names[int(order[0])],'margin':float(scores[ranked[0]]-scores[ranked[1]]),'top_k':[{'label':self.labels[int(i)],'name':self.names[int(i)],'score':float(scores[i])} for i in order]}

