import torch
from torch import nn

'''
class PositionEncoder
    sin cos
'''
class PositionalEncoder(nn.Module):
    def __init__(self, d_input, n_freqs,log_space = False ):
        super(PositionalEncoder, self).__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        self.freq = freq_bands
        

    def forward(self, x):
        res = [x]
        for freq in self.freq:
            res.append(self.sin_fn(x,freq))
            res.append(self.cos_fn(x,freq))
        return torch.cat(res, dim=-1)
    
    def sin_fn(self,x, freq):
        return torch.sin(x * freq)

    def cos_fn(self,x, freq):
        return torch.cos(x * freq)
