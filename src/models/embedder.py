import torch 
from torch import nn 
from torch.nn.utils.rnn import pad_packed_sequence

from .utils import packed_cat, repack

class Time2Vec(nn.Module):
    '''
    Recreating Time2Vec: Learning a Vector Representation of Time
        https://arxiv.org/abs/1907.05321
    '''
    def __init__(self, dim, is_sin=True):
        super().__init__()
        assert dim > 1, \
            'Must have at least 1 periodic feature, and 1 linear (dim must be >= 2)'
        
        self.lin = nn.Linear(1, dim)
        self.f = torch.sin if is_sin else torch.cos

    def forward(self, times):
        x = times.data 
        x = self.lin(x)
        periodic = self.f(x[:, 1:])
        x = torch.cat([x[:, 0].unsqueeze(-1), periodic], dim=1)
        return repack(x, times)
        

class KQV(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.kqv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_feats, out_feats),
                    nn.ReLU()
                )
            for _ in range(3)
        ])

    def forward(self, x):
        return [f(x) for f in self.kqv]

class Attention(nn.Module):
    '''
    Using builtin torch nn.Multihead attention function 
    Because some samples have so many values, iterate through them 
    as a list and only do one at a time, otherwise will quickly run out
    of memory 
    '''
    def __init__(self, in_feats, hidden, out, heads=8, layers=4):
        super().__init__() 

        self.layers = layers

        self.kqvs = nn.ModuleList(
            [KQV(in_feats, hidden)] +   
            [KQV(hidden, hidden) for _ in range(layers-1)]
        )

        self.attns = nn.ModuleList(
            [nn.MultiheadAttention(
                hidden, heads, dropout=0.25
            ) for _ in range(layers)]
        )

        self.project = nn.Sequential(
            nn.Linear(hidden, out),
            nn.ReLU()
        )

    def forward(self, ts, x, batch_size=None):
        x = packed_cat([ts, x])
        x,seq_len = pad_packed_sequence(x)

        if batch_size is None:
            batch_size = x.size(1)

        outs = []
        for i in range((x.size(0)//batch_size)+1):
            seq = x[:, i:i+batch_size, :]
            
            mask = torch.zeros((seq.size(1), seq.size(0)))
            for j in range(seq.size(1)):
                mask[j, seq_len[i+j]:] = 1

            for l in range(self.layers):
                q,k,v = self.kqvs[l](seq)
                seq,_ = self.attns[l](q,k,v, key_padding_mask=mask)

            sizes = seq_len[i:i+batch_size]
            outs.append(seq.sum(dim=0).div(sizes.unsqueeze(-1)))
        
        return self.project(torch.cat(outs, dim=0))


class NodeEmbedder(nn.Module):
    def __init__(self, f_feats, r_feats, hidden, out, embed_size, t2v_dim=8, attn_kw={}):
        super().__init__()

        self.f_t2v = Time2Vec(t2v_dim)
        self.f_attn = Attention(f_feats+t2v_dim, hidden, out, **attn_kw)
        
        self.r_t2v = Time2Vec(t2v_dim)
        self.r_attn = Attention(r_feats+t2v_dim, hidden, out, **attn_kw)

        self.combo = nn.Linear(out*2, embed_size)

    def forward(self, data):        
        t,f = data['files']
        f = self.f_attn(self.f_t2v(t), f)

        t,r = data['regs']
        r = self.r_attn(self.r_t2v(t), r)

        x = torch.cat([f,r], dim=1)
        return torch.sigmoid(self.combo(x))