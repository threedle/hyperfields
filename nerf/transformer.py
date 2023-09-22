import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from pdb import set_trace
#from .models import register


class Attention(nn.Module):

    def __init__(self, dim, n_head, head_dim, dropout=0.):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1) # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



def adaptive_norm(x,condition_vec):
    set_trace()
    style_mean = condition_vec.mean(dim=-1)
    style_std  = condition_vec.std(dim=-1)

    content_mean = x.mean(dim=-1)
    content_std  = x.std(dim=-1)

    normalized_feat = (x - content_mean.unsqueeze(-1))/content_std.unsqueeze(-1)
    out = (normalized_feat * style_std) +  style_mean
    return out


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, cond_vec):
        #set_trace()
        return self.fn(self.norm(x))
        #return self.fn(adaptive_norm(x,cond_vec))




#@register('transformer_encoder')
class TransformerEncoder(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0., condition_trans = False):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.condition_trans =  condition_trans
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x,data):
        data = data.unsqueeze(0)

        layer_id = 0
        for norm_attn, norm_ff in self.layers:
            if layer_id != 0 and self.condition_trans :
                x = torch.cat([data.repeat(1,10,1),x],dim=1)
            x = x + norm_attn(x,data)
            x = x + norm_ff(x,data)
            layer_id = layer_id+1
        return x
