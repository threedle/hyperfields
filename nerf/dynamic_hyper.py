import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
#from .models import *
#import .models
#from .models import register
from pdb import set_trace

def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()


#@register('trans_inr')
class DyTransInr(nn.Module):

    def __init__(self,hyponet_param_shapes, n_groups, transformer_encoder):
        super().__init__()
        dim = transformer_encoder.dim
        self.hyponet_param_shapes = hyponet_param_shapes
        self.transformer_encoder = transformer_encoder
        #self.tokenizer = models.make(tokenizer, args={'dim': dim})
        #self.hyponet = make(hyponet)
        #self.transformer_encoder = make(transformer_encoder)
        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        self.hypo_meta_data = [ item  for item in list(self.hyponet_param_shapes)]
        act_shapes = [ item[1][0]  for item in list(self.hyponet_param_shapes)]
        act_shapes.insert(0,0)
        
        for idx, (name, shape) in enumerate(self.hyponet_param_shapes):
            self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(shape[1], shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim+act_shapes[idx]),
                nn.Linear(dim+act_shapes[idx], dim+act_shapes[idx] ),
                nn.ReLU(),
                nn.Linear(dim+act_shapes[idx],np.prod(shape))
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

    def get_scene_vec(self, data ):
        #TODO change tokens
        dtokens = data #self.tokenizer(data)
        B = 1#dtokens.shape[0]
        dtokens = einops.repeat(dtokens, 'n d -> b n d', b=B)
        #dtokens = dtokens.repeat(1,50,1)
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        #trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1), data)
        #trans_out = trans_out[:, -len(self.wtokens):, :]
        trans_out = self.transformer_encoder(dtokens, data)
        cond_vec = trans_out.mean(dim=1)
        return cond_vec

    def get_params(self,cond_vec, index, prev_act = None):
        if prev_act is not None:
            prev_act = prev_act.mean(dim=0).unsqueeze(0) # mean over the rays and points sampled, its ok to do this as we want to condition the MLP on the basis of the scene 
            inp = torch.cat((cond_vec,prev_act),dim=1)
        else:
            inp = cond_vec
        params = dict()
        B=1
        #for name, shape in self.hyponet_param_shapes:
        name, shape = self.hypo_meta_data[index]
        wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
        #w, b = wb[:, :-1, :], wb[:, -1:, :]
        w = wb
        l, r = self.wtoken_rng[name]
        #x = self.wtoken_postfc[name](trans_out[:, l: r, :])
        x = self.wtoken_postfc[name](inp.unsqueeze(1))
        #set_trace()
        x = x.view(*shape).unsqueeze(0)
        #set_trace()
        #x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
        w = F.normalize(w * x, dim=1)

        wb = w#torch.cat([w, b], dim=1)
        #set_trace()
        params[name] = wb.squeeze(0)

        
        #set_trace()
        return params
         
    def forward(self, data):
        #TODO change tokens
        set_trace()
        dtokens = data #self.tokenizer(data)
        B = 1#dtokens.shape[0]
        dtokens = einops.repeat(dtokens, 'n d -> b n d', b=B)
        #dtokens = dtokens.repeat(1,50,1)
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        #trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1), data)
        #trans_out = trans_out[:, -len(self.wtokens):, :]
        trans_out = self.transformer_encoder(dtokens, data)
        cond_vec = trans_out.mean(dim=1)
        
        params = dict()
        for name, shape in self.hyponet_param_shapes:
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            #w, b = wb[:, :-1, :], wb[:, -1:, :]
            w = wb
            l, r = self.wtoken_rng[name]
            #x = self.wtoken_postfc[name](trans_out[:, l: r, :])
            x = self.wtoken_postfc[name](cond_vec.unsqueeze(1))
            #set_trace()
            x = x.view(*shape).unsqueeze(0)
            #set_trace()
            #x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            w = F.normalize(w * x, dim=1)

            wb = w#torch.cat([w, b], dim=1)
            #set_trace()    
            params[name] = wb.squeeze(0)
        #set_trace()
        return params
        #self.hyponet.set_params(params)
        #return self.hyponet

