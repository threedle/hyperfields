import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils_eval import * 
from optimizer import Shampoo
import wandb
from pdb import set_trace
import copy
import shutil
torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True
# from nerf.gui import NeRFGUI

# torch.autograd.set_detect_anomaly(True)
def clear_directory(path):
    import shutil
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, nargs="+", default=None, help="text prompt")
    parser.add_argument('--teacher_text', type=str, nargs="+", default=None, help="text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=10000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=64, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    # model options
    parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [grid, tcnn, vanilla]")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true', help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_stable_diff', type=float, default=1, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_teacher_image', type=float, default=1, help="loss scale for alpha value")
    parser.add_argument('--lambda_rgb', type=float, default=1, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_sigma', type=float, default=1, help="loss scale for alpha value")
    parser.add_argument('--lambda_depth', type=float, default=1, help="loss scale for alpha entropy")

    ### stable training options
    parser.add_argument('--clip_grad', action='store_true', help="overwrite current experiment")
    parser.add_argument('--fine_tune_conditioner', action='store_true', help="overwrite current experiment")
    parser.add_argument('--clip_grad_val', default = 1.0, type=float, help="overwrite current experiment")
    parser.add_argument('--init', default = None)
    parser.add_argument('--normalization', type = str, default = 'No')
    parser.add_argument('--WN', type = str, default = None)
    # ### GUI options
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")

    ### Logging options
    parser.add_argument('--wandb_flag', action='store_true', help="log in wandb")
    parser.add_argument('--project_name', type=str, default='test')    
    parser.add_argument('--exp_name', type=str, default='test')    
    parser.add_argument('--overwrite', action='store_true', help="overwrite current experiment")

    ###Network options
    parser.add_argument('--num_layers', type=int, default=3, help="render width for NeRF in training")
    parser.add_argument('--hidden_dim', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--skip', action = 'store_true')
    parser.add_argument('--bottleneck', action = 'store_true')
    parser.add_argument('--arch', type = str, default='mlp')
    ### Conditioning options
    parser.add_argument('--conditioning_model', type=str, default=None)
    parser.add_argument('--conditioning_mode', type=str, default='cat')
    parser.add_argument('--conditioning_dim', type = int, default = 64 )
    parser.add_argument('--meta_batch_size', type = int, default = 1)
    parser.add_argument('--multiple_conditioning_transformers', action = 'store_true')
    parser.add_argument('--condition_trans', action = 'store_true')
    parser.add_argument('--phrasing', action = 'store_true')
    parser.add_argument('--curricullum', action = 'store_true')

    ### Distillation options
    parser.add_argument('--load_teachers', type=str, default=None)
    #parser.add_argument('--teacher_size', nargs='+', type = int, default = None ) 
    parser.add_argument('--teacher_size', type = int, default = None )
    #### Other option
    parser.add_argument('--mem', action='store_true', help="overwrite current experiment")
    parser.add_argument('--dummy', action='store_true', help="overwrite current experiment")
    parser.add_argument('--test_teachers', action='store_true', help="overwrite current experiment")
    parser.add_argument('--not_diff_loss', action='store_true', help="overwrite current experiment") 
    parser.add_argument('--dist_image_loss', action='store_true', help="overwrite current experiment")
    parser.add_argument('--dist_sigma_rgb_loss', action='store_true', help="overwrite current experiment")
    parser.add_argument('--dist_depth_loss', action='store_true', help="overwrite current experiment")
    parser.add_argument('--skip_list', nargs='+', type = int, default = None)
    parser.add_argument('--train_list', nargs='+', type = int, default = None)
    parser.add_argument('--test_list', nargs='+', type = int, default = None)
    parser.add_argument('--pre_trained_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default='train', help="run in train or eval mdoe")
    # parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    # parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
    # parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    # parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    # parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    opt = parser.parse_args()
    if opt.arch != 'mlp':
        
        with open(opt.text[0]) as f:
            lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines]
        opt.text = lines

    if opt.teacher_text:
        with open(opt.teacher_text[0]) as f:
                lines = f.readlines()
        opt.teacher_text = [" ".join(line.split()) for line in lines]
  
    opt.num_scenes = len(opt.text)
    opt.workspace = os.path.join("outputs", opt.project_name+'_'+opt.exp_name)
    print(opt.text)
    if opt.pre_trained_path is not None:
        os.makedirs(opt.workspace+"/checkpoints")
        print('copying...')
        shutil.copyfile(sorted(glob.glob(f'{opt.pre_trained_path}checkpoints/*.pth'))[-1], f'{opt.workspace}/checkpoints/df_ep0001.pth') 
    if opt.overwrite and os.path.exists(opt.workspace): 
        clear_directory(opt.workspace)
       
    if opt.wandb_flag:
        resume_flag = opt.ckpt == 'latest'
        wandb.init(project = opt.project_name,config = opt, resume = True, name = opt.exp_name, id = opt.exp_name)
    else:
        wandb = None
    if opt.O:
        opt.fp16 = False
        opt.dir_text = True
        # use occupancy grid to prune ray sampling, faster rendering.
        opt.cuda_ray = True
        # opt.lambda_entropy = 1e-4
        # opt.lambda_opacity = 0

    elif opt.O2:
        opt.fp16 = True
        opt.dir_text = True
        opt.lambda_entropy = 1e-4 # necessary to keep non-empty
        opt.lambda_opacity = 3e-3 # no occupancy grid, so use a stronger opacity loss.

    '''
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'tcnn':
        from nerf.network_tcnn import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')
    '''
    print(opt)
   
    seed_everything(opt.seed)
    '''
    if  'hyper_transformer' in opt.arch:
       if 'split' in opt.arch:
           from nerf.split_hyper_network_grid import HyperTransNeRFNetwork as NeRFNetwork
       else:
    '''
    from nerf.hyper_network_grid import HyperTransNeRFNetwork as NeRFNetwork
    model = nn.DataParallel(NeRFNetwork(opt, num_layers= opt.num_layers, hidden_dim = opt.hidden_dim,wandb_obj=wandb ), device_ids = [0])

    '''
    if True:
        model = nn.DataParallel(NeRFNetwork(opt, num_layers= opt.num_layers, hidden_dim = opt.hidden_dim,wandb_obj=wandb ), device_ids = [0])
    else:
        model = NeRFNetwork(opt, num_layers= opt.num_layers, hidden_dim = opt.hidden_dim)
    '''

    if opt.load_teachers is not None: 
        with open(opt.load_teachers) as f:
            lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines]
        opt.teacher_paths = lines

        model.teacher_models = []
        for idx, path in enumerate(opt.teacher_paths):
            print(path)
            model_path =  glob.glob(opt.teacher_paths[idx]+"/checkpoints/*")[-1]
            model_teacher = nn.DataParallel(NeRFNetwork(opt, num_layers= opt.num_layers, hidden_dim = opt.hidden_dim,wandb_obj=wandb, teacher_flag = True, teacher_id = idx), device_ids = [0])
        #TODO fix these
            
            #model_teacher.module.scene_id = idx
            model_teacher.module.sigma_net.epoch = 0
            model_teacher.module.load_checkpoint( checkpoint = model_path)
            model.teacher_models.append(model_teacher)
            if opt.test_teachers:
                for index in range(opt.num_scenes):
                    model_teacher.module.scene_id = index % opt.teacher_size
                    #model_teacher.module.scene_id = idx
                    model_teacher.module.load_checkpoint( checkpoint = model_path)
                    from nerf.sd import StableDiffusion
                    guidance = StableDiffusion('cuda')
                    trainer = Trainer('df', opt, model_teacher, guidance, device='cuda', workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='scratch')
                    test_loader = NeRFDataset(opt, device='cuda', type='test', H=opt.H, W=opt.W, size=100).dataloader()
                    model_teacher.module.sigma_net.epoch = 0
                    trainer.test(test_loader, scene_id = idx)
    

     
    #print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
        trainer.test(test_loader)
        
        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
    
    else:
        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion
            guidance = StableDiffusion(device)
        elif opt.guidance == 'clip':
            from nerf.clip import CLIP
            guidance = CLIP(device)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        optimizer = lambda model: torch.optim.Adam(model.module.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        #optimizer = lambda model: Adan(model.module.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        # optimizer = lambda model: Shampoo(model.get_params(opt.lr))

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1e-3 if iter < 100 else  0.1 ** min(iter / opt.iters, 1))
        #scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter:  0.1 ** min(iter / opt.iters, 1))
        # scheduler = lambda optimizer: optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=opt.iters, pct_start=0.1)

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True, wandb_obj = wandb)

        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

        trainer.get_results(train_loader, valid_loader,test_loader, max_epoch)
        # also test
        trainer.test(test_loader)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)
