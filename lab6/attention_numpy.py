"""Stable NumPy scaled dot-product attention; no framework attention API."""
import numpy as np

def stable_softmax(x,axis=-1):
    x=np.asarray(x,dtype=np.float64)
    if x.size==0 or np.isnan(x).any() or np.isposinf(x).any():
        raise ValueError("Softmax requires nonempty scores without NaN/+inf")
    maximum=np.max(x,axis=axis,keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("Every row needs at least one unmasked finite score")
    exponent=np.exp(x-maximum)
    return exponent/exponent.sum(axis=axis,keepdims=True)

def scaled_dot_product_attention(Q,K,V,temperature=1.0,mask=None,scaled=True):
    """Rows are queries/keys. Return scores [nq,nk], weights [nq,nk], context [nq,dv].
    mask=True means allowed; fully masked rows raise ValueError.
    scaled=False is an explicit dot-product ablation.
    """
    Q,K,V=[np.asarray(x,dtype=np.float64) for x in (Q,K,V)]
    if any(x.ndim!=2 for x in (Q,K,V)):
        raise ValueError("Q, K and V must be 2D matrices")
    if Q.shape[1]!=K.shape[1] or K.shape[0]!=V.shape[0] or Q.shape[1]==0:
        raise ValueError("Incompatible query/key/value dimensions")
    if not all(np.isfinite(x).all() for x in (Q,K,V)):
        raise ValueError("Q/K/V must be finite")
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError("Temperature must be positive")
    scores=Q@K.T
    scores=scores/(np.sqrt(Q.shape[1]) if scaled else 1)/temperature
    if mask is not None:
        allowed=np.broadcast_to(np.asarray(mask,dtype=bool),scores.shape)
        scores=np.where(allowed,scores,-np.inf)
    weights=stable_softmax(scores,axis=-1)
    return scores,weights,weights@V

def entropy(weights,axis=-1):
    weights=np.asarray(weights,dtype=float)
    return -(weights*np.log(np.clip(weights,1e-300,1))).sum(axis=axis)
