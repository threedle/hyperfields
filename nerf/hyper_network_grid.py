import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize
from pdb import set_trace
import clip
import GPUtil
import gc
from torch.nn.utils import weight_norm as wn
import math

from .transformer import TransformerEncoder
from .dynamic_hyper import DyTransInr
from .layers import batched_linear_mm
import glob
import copy

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class PositionalEncoding(nn.Module):

    def __init__(self, d_model = 32, max_len= 6):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, nerf_conditioning = False, bias=True, init= None, opt= None, wandb_obj = None, hyper_flag = False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.nerf_conditioning = nerf_conditioning
        self.init = init
        self.opt = opt
        self.wandb_obj = wandb_obj
        self.hyper_flag =  hyper_flag
        net = []


        if self.nerf_conditioning:
            self.transformer_encoder = TransformerEncoder(256, 6,12,16,64, condition_trans = opt.condition_trans)
            self.hyper_transform = nn.Linear(768,256)
            nn.init.orthogonal_(self.hyper_transform.weight)

            transform_dim = opt.conditioning_dim 

            #if opt.multiple_conditioning_transformers:
            self.transform_list = nn.ModuleList()
            for i in range(num_layers):
                self.transform_list.append(nn.Sequential(wn(nn.Linear(768, self.dim_hidden *2)), nn.ReLU(), wn(nn.Linear(self.dim_hidden*2, transform_dim)))) 
                self.apply_init(model = self.transform_list[i], init='ortho')


            #Creating a dummy MLP to fill its weights up
            for l in range(num_layers):
                inp_dim_aug_dim = 0 
                if l!=0:
                         
                    inp_dim_aug_dim = inp_dim_aug_dim + 32 # include positional conditioning 

                if  l<=4:
                    inp_dim_aug_dim = inp_dim_aug_dim + transform_dim #include scene conditioning
                                
                net.append(nn.Linear(self.dim_in + inp_dim_aug_dim if l == 0 else self.dim_hidden + inp_dim_aug_dim, self.dim_out if l == num_layers - 1 else self.dim_hidden,bias=bias))

            param_shapes= {}
            for idx, layer in enumerate(net):
                param_shapes['layer_{}'.format(idx)] = layer.weight.shape
           
            self.hyper_transformer = DyTransInr(param_shapes.items(), self.dim_hidden, self.transformer_encoder)
        else:
            for l in range(num_layers):
                net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden,bias=bias))

            self.net = nn.ModuleList(net)
            if self.init is not None:
                set_trace()
                self.apply_init()

    def load_teacher(self,opt, checkpoint=None, model_only=True, teacher_id = None):
        #if teacher_id is None:

        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.module.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.module.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.module.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

        

    def apply_init(self, model =None,init =None):
        if model is None:
            model = self.net
        if init is None:
            init = self.init

        for ind, layer_ in enumerate(model):
            if type(layer_) == type(nn.ReLU()):
                continue
            elif init == 'ortho':
                if ind <self.num_layers -1:
                    nn.init.orthogonal_(layer_.weight)
                    if self.opt.WN :
                        if not ('not_first' in self.opt.WN and ind ==0): 
                            layer_.weight_g.data.fill_(1.0) 

            else:
                print('maintain default')
            #print('checking')
    def adaptive_norm(self,x,condition_vec):
        style_mean = condition_vec.mean()
        style_std  = condition_vec.std()
        
        content_mean = x.mean(dim=-1)
        content_std  = x.std(dim=-1)
        
        normalized_feat = (x - content_mean.unsqueeze(-1))/content_std.unsqueeze(-1)
        out = (normalized_feat * style_std) +  style_mean
        return out
        
    def forward(self, x, conditioning_vector = None,epoch = None):
        #print(x.device)i
        x = x/x.norm(dim=1).unsqueeze(dim=1) #* 10

        pos_enc = x

        if conditioning_vector is not None:
            scene_id = self.scene_id
        #if epoch ==11:
        if self.nerf_conditioning:# and conditioning_vector is not None:
            processed_tokens = self.hyper_transform(conditioning_vector[scene_id]['input_tokens'].squeeze(0))
            processed_scene_vec = self.hyper_transformer.get_scene_vec(processed_tokens) 
         
            #self.set_params(params)
        hyper_inp = None
        for l in range(self.num_layers):
            # skip options                
            #if l % 2 ==1 and l>1 and self.opt is not None and self.opt.skip:
            #    x = x + x_skip 

            #Pre-Normalization
            # conditioning options
            if self.nerf_conditioning  and l<=4 :

                #x = torch.cat((x+self.transform(conditioning_vector['input_vec']).repeat(x.shape[0],1)), dim=1)
                #if self.opt.multiple_conditioning_transformers:
                transformer = self.transform_list[l]
                proj_cond_vec = transformer(conditioning_vector[scene_id]['input_vec'])

                proj_cond_vec = torch.nn.functional.layer_norm(proj_cond_vec, (proj_cond_vec.shape[1],))
                proj_cond_vec = proj_cond_vec /proj_cond_vec.norm().detach()
                x = torch.cat((x,proj_cond_vec.repeat(x.shape[0],1)), dim=1)


            if self.nerf_conditioning and l!=0:
                x = torch.cat((x,pos_enc), dim=1)

            if self.nerf_conditioning:
                # forward pass
                #call MLPs to generate weights of the subsquent layer
                #processed_scene_vec is the conditioning token
                #l is layer id, used to index the correct MLP
                #hyper_inp is the activation of the previous layer used as input to the current layer
                params = self.hyper_transformer.get_params(processed_scene_vec, l, hyper_inp) #call MLPs to generate weights of the subsquent layer
                x = F.linear(x,params['layer_{}'.format(l)])#batched_linear_mm(x, params['layer_{}'.format(l)])i
                if 'detach'  in self.opt.arch:
                    hyper_inp = x.detach()
                else:
                    hyper_inp = x
            else:
                x = self.net[l](x)

            #Post Normalization
            if self.opt is not None and l != self.num_layers - 1 and self.nerf_conditioning: 
                x = self.adaptive_norm(x,conditioning_vector[scene_id]['input_vec'])

            # skip options
            if l % 2 ==0 and l>1 and self.opt is not None and self.opt.skip and l != self.num_layers-1:
                x = x + x_skip

            # Activation
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
            
            # store for skip 
            if l %2 == 0:
                x_skip = x
        
        return x


class HyperTransNeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=3,
                 hidden_dim=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 wandb_obj = None,
                 teacher_flag = False,
                 teacher_id = None
                 ):
        
        super().__init__(opt, teacher_flag, teacher_id)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.teacher_flag = teacher_flag

        self.conditioning_model = opt.conditioning_model
        if opt.conditioning_model == 'CLIP':
            self.clip_model, clip_preprocess = clip.load("ViT-B/16", device='cuda', jit=False)
            for parameters in self.clip_model.parameters():
                parameters.requires_grad = False
            self.nerf_conditioning = True
        elif opt.conditioning_model == 'bert':
            from transformers import AutoTokenizer, AutoModel
            self.text_model_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', cache_dir = "./local_dir")
            self.text_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens',cache_dir = "./local_dir").cuda()
            for parameters in list(self.text_model.parameters()):
                parameters.requires_grad = False

            if self.opt.fine_tune_conditioner:
                for parameters in list(self.text_model.parameters())[-34:-2]:
                    parameters.requires_grad = True
                    
            self.nerf_conditioning = True

        elif opt.conditioning_model == 'T5':
            #from transformers import T5Tokenizer, T5EncoderModel, T5Model

            #self.text_model_tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir = "./local_cache")
            #self.text_model = T5Model.from_pretrained("t5-small", cache_dir = "./local_cache")

            #input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            #outputs = model(input_ids=input_ids)
            #last_hidden_states = outputs.last_hidden_state

            from transformers import AutoTokenizer, AutoModelWithLMHead
            import os
            os.environ['TRANSFORMERS_CACHE'] = './cache'
            self.text_model_tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir = "./cache")
            self.text_model = AutoModelWithLMHead.from_pretrained("t5-small", cache_dir = "./cache").cuda()
            for parameters in self.text_model.parameters():
                parameters.requires_grad = True
            self.nerf_conditioning = True
 
        elif opt.conditioning_model is None:
            self.nerf_conditioning = False 

        if self.nerf_conditioning:
            #with torch.no_grad():
                
            self.conditioning_vector = {}
            for idx, val in enumerate(self.opt.text):
               current_emb  = self.get_conditioning_vec(idx)
               self.conditioning_vector[idx] = current_emb
            #del self.text_model_tokenizer
            #self.text_model_tokenizer = None
            #gc.collect()
            #torch.cuda.empty_cache()
        else:
            self.conditioning_vector = None

        self.encoder, self.in_dim = get_encoder('tiledgrid', input_dim=3, desired_resolution=2048 * self.bound)
        

        self.sigma_net = nn.DataParallel(MLP(self.in_dim, 4, hidden_dim, num_layers, self.nerf_conditioning,bias=True, init= opt.init, opt = opt, wandb_obj = wandb_obj, hyper_flag = True))
        #self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, self.nerf_conditioning,bias=True)
        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            # self.encoder_bg, self.in_dim_bg = get_encoder('tiledgrid', input_dim=2, num_levels=4, desired_resolution=2048)
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)

            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None



    def load_checkpoint(self, checkpoint=None, model_only=True):

        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, 'cuda')
       
        checkpoint_dict['model_renamed']  = {}

        for element in checkpoint_dict['model'].keys():
            new_key = '.'.join(element.split('.')[1:])
            checkpoint_dict['model_renamed'][new_key] =  checkpoint_dict['model'][element] 


        if 'model' not in checkpoint_dict:
            self.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint_dict['model_renamed'], strict=False)
        
        print("[INFO] loaded teacher model.")
        if len(missing_keys) > 0:
            set_trace()
            print(f"[WARN] missing keys: {missing_keys}")

        if len(unexpected_keys) > 0:
            set_trace()
            print(f"[WARN] unexpected keys: {unexpected_keys}")

        if model_only:
            return


    def get_conditioning_vec(self,index=0):
        conditioning_vector = None
 
        if self.conditioning_model == 'CLIP':
            ref_text = self.opt.text[index]
            conditioning_vector = {}
            #if self.opt.dir_text:
            #    print('not implemented')
            text_token = clip.tokenize(ref_text).to('cuda')
            with torch.no_grad():
                conditioning_tokens = self.clip_model.encode_text(text_token)
            conditioning_tokens = conditioning_tokens/ conditioning_tokens.norm(dim=-1, keepdim =True)
            conditioning_vector['input_vec']  = conditioning_tokens
            conditioning_vector['input_tokens'] = conditioning_tokens

        elif self.conditioning_model == 'T5':
            ref_text = self.opt.text[index]
            conditioning_vector = {}
            #if self.opt.dir_text:
            #    print('not implemented')
            #self.text_model.eval()
            text_token =torch.tensor(self.text_model_tokenizer([ref_text])['input_ids']).to('cuda')
            decoder_input_ids = self.text_model_tokenizer("dummy text to be ignored", return_tensors="pt").input_ids.cuda()
            self.text_token = text_token
            self.decoder_input_ids = decoder_input_ids
            if False: #TODO change later sud
                conditioning_tokens = self.text_model(text_token, decoder_input_ids = decoder_input_ids)['encoder_last_hidden_state']
            else:
                with torch.no_grad():
                    conditioning_tokens = self.text_model(text_token, decoder_input_ids = decoder_input_ids)['encoder_last_hidden_state']
            conditioning_vector['input_vec'] = conditioning_tokens.mean(dim=1)/conditioning_tokens.mean(dim=1).norm(dim=-1, keepdim=True)
            conditioning_vector['input_tokens'] = conditioning_tokens/conditioning_tokens.norm(dim=-1, keepdim = True )
        
        elif self.conditioning_model == 'bert':
            
            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output[0] #First element of model_output contains all token embeddings
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            #if self.teacher_flag:
            #    ref_text = self.opt.text[index]
            #else:
            #    ref_text = self.opt.teacher_text[index]

            ref_text = self.opt.text[index]
            conditioning_vector = {}     
             
            text_token = self.text_model_tokenizer(ref_text, padding=True, truncation=True, return_tensors='pt')
            text_token['input_ids'] = text_token['input_ids'].cuda()
            text_token['token_type_ids'] = text_token['token_type_ids'].cuda()
            text_token['attention_mask'] = text_token['attention_mask'].cuda()
            if self.opt.fine_tune_conditioner:
                model_output = self.text_model(**text_token)
            else:
                with torch.no_grad():
                    model_output = self.text_model(**text_token)
            conditioning_tokens = model_output[0]
            
            inp_vec = mean_pooling(model_output, text_token['attention_mask'])
            conditioning_vector['input_vec']  = inp_vec/inp_vec.norm(dim=-1, keepdim=True)
            conditioning_vector['input_tokens'] = conditioning_tokens/conditioning_tokens.norm(dim=-1, keepdim = True )           

        return conditioning_vector    

    '''
    def __getattr__(self,name):
        if name in [self.get_conditioning_vec]:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)
    '''
    # add a density blob to the scene center
    def gaussian(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g
       

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        if self.opt.mem:
            torch.cuda.empty_cache()
            gc.collect() 
        h = self.encoder(x, bound=self.bound)

        #cur_mem = torch.cuda.memory_allocated() * 1e-9 
        #max_mem = torch.cuda.max_memory_allocated() * 1e-9 
        #print(cur_mem/max_mem)
        #print(h.shape)
        #if self.sigma_net.epoch == 11:
        self.sigma_net.module.scene_id = self.scene_id 
        temp = self.get_conditioning_vec(index = self.scene_id)
 
        self.conditioning_vector[self.scene_id]  = temp
        h = self.sigma_net(h, conditioning_vector = self.conditioning_vector, epoch = self.sigma_net.epoch)

        sigma = trunc_exp(h[..., 0] + self.gaussian(x))
        albedo = torch.sigmoid(h[..., 1:])
        if self.opt.mem:
            torch.cuda.empty_cache()
            gc.collect() 
        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return normal
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        if shading == 'albedo':
            # no need to query normal
            sigma, color = self.common_forward(x)
            normal = None
        
        else:
            # query normal
            sigma, albedo = self.common_forward(x)
            normal = self.finite_difference_normal(x)

            # with torch.enable_grad():
            #     x.requires_grad_(True)
            #     sigma, albedo = self.common_forward(x)
            #     # query gradient
            #     normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]

            # normalize...
            normal = safe_normalize(normal)
            normal[torch.isnan(normal)] = 0

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ -l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
        
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):
        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': list(self.text_model.parameters())[-34: -2], 'lr': lr }
        ]        

        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params
