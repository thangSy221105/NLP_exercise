"""From-scratch NumPy RNN cell: h_t=tanh(W_xh x_t + W_hh h_prev + b_h)."""
import numpy as np
class SimpleRNNCell:
    def __init__(self,input_size,hidden_size,seed=2026):
        if input_size<1 or hidden_size<1:
            raise ValueError("Sizes must be positive")
        rng=np.random.default_rng(seed)
        self.W_xh=rng.normal(0,1/np.sqrt(input_size),(hidden_size,input_size))
        self.W_hh=rng.normal(0,1/np.sqrt(hidden_size),(hidden_size,hidden_size))
        self.b_h=np.zeros(hidden_size)
    def forward(self,x_t,h_prev):
        x_t=np.asarray(x_t,dtype=float)
        h_prev=np.asarray(h_prev,dtype=float)
        if x_t.shape!=(self.W_xh.shape[1],) or h_prev.shape!=(self.W_hh.shape[1],):
            raise ValueError("Input/hidden vector shape mismatch")
        return np.tanh(self.W_xh@x_t+self.W_hh@h_prev+self.b_h)
