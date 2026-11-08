from __future__ import annotations
import unicodedata
import numpy as np

_BOS, _EOS, _WORD_START, _WORD_END = 0x110000, 0x110001, 0x110002, 0x110003

def normalize_text(text: str, *, casefold: bool) -> str:
    text = unicodedata.normalize('NFC', text)
    if casefold: text = unicodedata.normalize('NFC', text.casefold())
    kept, in_space = [], True
    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            if not in_space: kept.append(' ')
            in_space = True; continue
        if category in {'Cc','Cs','Co','Cn'}: continue
        if category == 'Cf' and char not in {'\u200c','\u200d'}: continue
        kept.append(char); in_space = False
    return ''.join(kept).strip()

def codepoint_stream(text: str, *, casefold: bool) -> np.ndarray:
    normalized = normalize_text(text, casefold=casefold); values = [_BOS]
    for token in normalized.split(' ') if normalized else ():
        values.extend((_WORD_START, *map(ord, token), _WORD_END))
    values.append(_EOS); return np.asarray(values, dtype=np.uint64)

def _hash_codes(codes, n, *, channel_salt):
    if len(codes) < n: return np.empty(0, dtype=np.uint64)
    size = len(codes)-n+1
    with np.errstate(over='ignore'):
        salt64 = (0xCBF29CE484222325 ^ channel_salt) & 0xFFFFFFFFFFFFFFFF
        hashes = np.full(size, np.uint64(salt64), dtype=np.uint64); multiplier=np.uint64(0x100000001B3)
        for offset in range(n): hashes = hashes*multiplier + codes[offset:offset+size] + np.uint64(1)
        hashes ^= hashes >> np.uint64(30); hashes *= np.uint64(0xBF58476D1CE4E5B9); hashes ^= hashes >> np.uint64(27); hashes *= np.uint64(0x94D049BB133111EB); hashes ^= hashes >> np.uint64(31)
    return hashes

def channels(text: str, max_n: int = 6):
    result=[]
    for view_index, casefold in enumerate((False, True)):
        codes=codepoint_stream(text, casefold=casefold)
        for n in range(1,max_n+1):
            salt=(view_index+1)*0x9E3779B185EBCA87 ^ n*0xC2B2AE3D27D4EB4F
            result.append(_hash_codes(codes,n,channel_salt=salt))
    return result

