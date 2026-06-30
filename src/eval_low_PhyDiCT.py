import argparse
import copy
import random
import torch
import numpy as np

from torch.utils import data
from pathlib import Path
from torch.optim import AdamW
from torchvision import transforms as T
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from dataloader import cache_transformed_text
import os

from accelerate import Accelerator

from utils import *
from train_low_res import Unet3D as UnetLR, EMA, unnormalize_img, normalize_img
from DiffDRR import Reconstruction
from train_low_res import GaussianDiffusion as GDLR
from taming.losses import LPIPS

from diffdrr.pose import convert

SEED = 1
random.seed(SEED)

# Seed NumPy
np.random.seed(SEED)

# Seed PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # If using multi-GPU setups

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OBSERVATION_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
xray_folder = 'YOUR X-RAY IMAGE FOLDER PATH'  # Replace with your actual path
add_time = 0.8 # EC step

class GDLR_reddiff(GDLR):
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)  # Pass all args/kwargs to Trainer
        self.lpips = LPIPS().cuda()
        print('If it the same source (DiffVox), you can use large lambda_MSE and small lambda_LPIPS, otherwise (DeepDRR) use larger lambda_LPIPS')
        self.lambda_MSE = 0.1
        self.lambda_LPIPS = 0.5


    def ec(self, x_start, cond,  cond_scale, indexes=None, use_ddim=True, noise=None,):
        device = cond['text'].device
        designed_add_time = torch.tensor([add_time])
        x_noisy, noise = get_z_t(x_start, designed_add_time, noise)

        if use_ddim:
            time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))
        else:
            time_steps = range(0, self.num_timesteps)

        img = x_noisy

        for i, t in enumerate(tqdm(reversed(time_steps), desc='sampling loop time step',
                                   total=len(time_steps))):
            
            if t > designed_add_time * self.num_timesteps:
                continue
            else:
                time = torch.full((1,), t, device=device, dtype=torch.float32)

                if use_ddim:
                    time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
                    img = self.p_sample_ddim(img, time, time_minus, indexes=indexes, cond=cond,
                                            cond_scale=cond_scale, ec=True)
                else:
                    img = self.p_sample(img, time, indexes=indexes, cond=cond,
                                        cond_scale=cond_scale)
        return unnormalize_img(img)
    

    def p_sample_ddim(self, x, t, t_minus, indexes=None, cond=None, cond_scale=1., clip_denoised=True, ec=False):
        with torch.no_grad():
            x_recon = self.denoise_fn.forward_with_cond_scale(x, t, indexes=indexes, cond=cond, cond_scale=cond_scale)
            if cond_scale != 1:
                x_recon, x_recon_null = x_recon
                eps = get_eps_x_t(x_recon, x, t)
                eps_null = get_eps_x_t(x_recon_null, x, t)
                final_eps = eps_null + (eps - eps_null) * cond_scale
                x_recon = get_x0_x_t(final_eps, x, t)

        x_recon_img = x_recon[:, 0:1].clone().detach()
        x_recon_img.requires_grad = True
        optimizer = torch.optim.Adam(
                            [x_recon_img], lr=1e-2  
                        )  
        if t.sum() and not ec:
            imgs = cond['cond']
            for l in range(2):
                x_recon_img_large = torch.nn.functional.interpolate(x_recon_img, size=256)
                style_losses = 0
                for idx, angle in enumerate(OBSERVATION_ANGLES):
                    recon = Reconstruction(x_recon_img_large.squeeze(), x_recon_img_large.device)
                    angle = torch.tensor(angle)/360
                    rot = torch.tensor([[0, angle.item()*np.pi*2, 0.0]], device=x_recon_img_large.device)
                    xyz = torch.tensor([[0.0, 950.0, 0.0]], device=x_recon_img_large.device)
                    pose = convert(rot, xyz, parameterization="euler_angles", convention="ZXY")
                    est = recon(pose)
                    print('IMPORTANT, PLEASE VISUALIZE THE ESTIMATED X-RAY IMAGE TO CHECK IF IT IS REASONABLE')
                    condition = est.contiguous().transpose(-1, -2).flip(-1)
                    condition = (condition - condition.mean())/condition.std()
                    style_losses += self.lambda_MSE*torch.nn.MSELoss()(condition[0], imgs[idx]) + self.lambda_LPIPS*self.lpips(condition[0].repeat(1, 3, 1, 1), imgs[idx].repeat(1, 3, 1, 1))
                txt_alignment_loss = torch.nn.functional.mse_loss(x_recon_img.clone().to('cuda'), x_recon[:, 0:1].clone().detach().to('cuda'))
                loss = (
                    1 * style_losses + 1 * txt_alignment_loss
                )

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
            x_recon[:, 0:1] = x_recon_img.detach().to('cuda')

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            x_recon = x_recon.clamp(-s, s) / s
        if t[0]<int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus, min=0.0)
            x = ddim_sample(x_recon, x, (t_minus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x 
        
    def p_sample_loop(self, shape, cond=None, cond_scale=1., use_ddim=True, noise=None):
        device = cond['cond'].device

        bsz = shape[0]

        if use_ddim:
            time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))
        else:
            time_steps = range(0, self.num_timesteps)
        if not noise:
            img = torch.randn(shape, device=device)
        else:
            img = torch.load(noise).to(device)
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        for i, t in enumerate(tqdm(reversed(time_steps), desc='sampling loop time step',
                                   total=len(time_steps))):
            time = torch.full((bsz,), t, device=device, dtype=torch.float32)
            # print('time', time)
            if use_ddim:
                time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
                img = self.p_sample_ddim(img, time, time_minus, indexes=indexes, cond=cond,
                                         cond_scale=cond_scale)
            else:
                img = self.p_sample(img, time, indexes=indexes, cond=cond,
                                    cond_scale=cond_scale)
        return unnormalize_img(img)

    def sample(self, cond=None, cond_scale=2., batch_size=16, DDIM=True, noise=None):
        print('you are here in the PhyDiCT algo')
        batch_size = cond['text'].shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond=cond,
                                      cond_scale=cond_scale, use_ddim=DDIM, noise=noise)


class Trainer(object):
    def __init__(
            self,
            diffusion_model_lr,
            folder,
            *,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            save_folder='',
            num_sample_rows=4,
            max_grad_norm=None
    ):
        super().__init__()
        self.model = diffusion_model_lr

        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = self.model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.save_folder = save_folder

        train_files = []

        self.text_folder = folder

        for img_dir in os.listdir(folder):
            if img_dir[-3:] == 'npy':
                train_files.append({'text': os.path.join(folder, img_dir)})

        self.ds = cache_transformed_text(train_files=train_files)

        print(f'found {len(self.ds)} videos as gif files at {folder}')
        assert len(self.ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

        self.dl = data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True)
        self.opt = AdamW(self.model.parameters(), lr=train_lr, betas=(0.9, 0.999))

        self.step = 0

        self.amp = amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

        if amp:
            mixed_precision = "fp16"
        else:
            mixed_precision = "fp32"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            mixed_precision=mixed_precision,
        )

        self.model, self.ema_model, self.dl, self.opt, self.step = self.accelerator.prepare(
            self.model, self.ema_model, self.dl, self.opt, self.step
        )
        

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        self.accelerator.save_state(str(self.results_folder / f'{milestone}_ckpt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            dirs = os.listdir(self.results_folder)
            # print('dirs', dirs)
            # dirs = [d for d in dirs if d.endswith("ckpt")]
            # dirs = sorted(dirs, key=lambda x: int(x.split("_")[0]))
            path = dirs[-1]

        self.step = 1
        pth = os.path.join(self.results_folder, path)
        self.accelerator.load_state(pth, strict=False)

    def phydict(self, *args, **kwargs):
        print("main Plug-and-Play PhyDiCT function for generating 3D volumes from text and X-ray images")

        for i, data in enumerate(self.dl):

            text = data["text"].squeeze(dim=1)
            text = text.to(self.accelerator.device)
            item  = data['text_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]

            file_name = item+'_sample_0.npy'
            if os.path.exists(os.path.join(self.save_folder, str(f'{file_name}'))):
                continue

            with torch.no_grad():
                all_condition = []
                all_angles = []
                for idx, angle in enumerate(OBSERVATION_ANGLES):
                    angle = torch.tensor(angle)/360
                    xray_path = os.path.join(xray_folder, item+f'.{idx}.png')
                    xray_img = Image.open(xray_path).convert("L")
                    arr = torch.from_numpy(np.array(xray_img).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                    arr = (arr-arr.mean()) / arr.std()
                    all_condition.append(arr)
                    all_angles.append(angle.unsqueeze(0))
                condition = torch.cat(all_condition, dim=0).to(self.accelerator.device)
                angle = torch.cat(all_angles, dim=0).to(self.accelerator.device)

            num_samples = self.num_sample_rows ** 2
            batches = num_to_groups(num_samples, self.batch_size)
            all_videos_list = list(
                map(lambda n: self.ema_model.sample(batch_size=n, cond={'text': text, 'cond': condition, 'angle': angle,}, noise=noise_time, cond_scale=10), batches))
            all_videos_list = torch.cat(all_videos_list, dim=0)
            np.save(os.path.join(self.save_folder, str(f'{file_name}')),
                    all_videos_list.cpu().numpy())
            

    def EC(self, *args, **kwargs):
        print("Extra Computing (EC) for better visual results")

        os.makedirs(os.path.join(self.save_folder, f'add_t_{add_time}'), exist_ok=True)
        volume = [item.split('_sample_0.npy')[0] for item in os.listdir(self.save_folder) if item.endswith('.npy')]

        for idx, item in enumerate(volume):
            file_name = item+'_sample_0.npy'
            if os.path.exists(os.path.join(self.save_folder, f'add_t_{add_time}', str(f'{file_name}'))):
                continue
            img = torch.from_numpy(np.load(os.path.join(self.save_folder, file_name))).to('cuda')
            img = normalize_img(img)

            text_name = item+'.npy'
            text = torch.from_numpy(np.load(os.path.join(self.text_folder, text_name))).cuda().unsqueeze(0)

            num_samples = self.num_sample_rows ** 2
            batches = num_to_groups(num_samples, self.batch_size)
            
            all_videos_list = list(
                map(lambda n: self.ema_model.ec(img, cond={'text': text,}, cond_scale=10, noise=noise_time,), batches))
            
            all_videos_list = torch.cat(all_videos_list, dim=0)
            np.save(os.path.join(self.save_folder, f'add_t_{add_time}', str(f'{file_name}')),
                    all_videos_list.cpu().numpy())


def main(args):
    
    model = UnetLR(
        dim=160,
        cond_dim=768,
        dim_mults=(1, 2, 4, 8),
        channels=4,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    )
    
    
    diffusion_model_lr = GDLR_reddiff(
        denoise_fn=model,
        image_size=64,
        num_frames=64,
        text_use_bert_cls=False,
        channels=4,
        timesteps=1000,
        loss_type='l2',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.995,
        volume_depth=64,
        ddim_timesteps=50,
    )
    

    trainer = Trainer(diffusion_model_lr=diffusion_model_lr,
                      folder=args.text_feature_folder,
                      ema_decay=0.995,
                      num_frames=64,
                      train_batch_size=1,
                      train_lr=1e-4,
                      train_num_steps=1000000,
                      gradient_accumulate_every=4,
                      amp=True,
                      step_start_ema=10000,
                      update_ema_every=10,
                      save_and_sample_every=1000,
                      results_folder=args.pretrain_model_path,
                      save_folder=args.save_path,
                      num_sample_rows=1,
                      num_sample=1,
                      max_grad_norm=1.0)
    
    trainer.load(-1)
    trainer.phydict()
    trainer.EC()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_feature_folder', type=str, default='./text_feature')
    parser.add_argument('--pretrain_model_path', type=str, default='./model/results_text_low_res_improved_unet_seg')
    parser.add_argument('--save_path', type=str, default='tmp/results_phydict_low_res')
    parser.add_argument('--noise_time', default=None)
    args = parser.parse_args()
    noise_time = args.noise_time

    os.makedirs(args.save_path, exist_ok=True)
    main(args)