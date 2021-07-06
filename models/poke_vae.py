import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
import wandb
import numpy as np
from tqdm import tqdm
from functools import partial
import os

from models.modules.motion_models.motion_encoder import resnet18,resnet18_alternative
from models.modules.discriminators.patchgan import PatchDiscriminator
from models.modules.discriminators.patchgan_3d import resnet
from models.modules.autoencoders.LPIPS import LPIPS as PerceptualLoss
from models.modules.autoencoders.fully_conv_models import ConvEncoder, SpadeCondConvDecoder
from models.modules.motion_models.rnn import ConvGRU
from lpips import LPIPS as lpips_net
from utils.metrics import LPIPS, SSIM_custom, PSNR_custom, FVD,calculate_FVD, compute_div_score, metric_vgg16, compute_div_score_mse, compute_div_score_lpips
from utils.losses import KL, VGGLoss
from utils.logging import make_video, make_flow_video_with_samples, make_samples_and_samplegrid, save_video
from utils.general import get_logger, linear_var


class PokeVAE(pl.LightningModule):

    def __init__(self,config, dirs):
        super().__init__()
        self.config = config
        self.test_mode = self.config['general']['test']
        #self.disc_start = self.config["training"]['pretrain']
        self.config["architecture"].update({"img_size":self.config["data"]["spatial_size"][0]})
        self.dirs = dirs

        # # features for fvd calculattion
        self.features_fvd_true = []
        self.features_fvd_fake = []
        self.fvd_features_true_x0 = []
        self.fvd_features_fake_x0 = []

        self.full_sequence = 'full_sequence' in self.config['training'] and self.config['training']['full_sequence']
        self.stack_motion_and_poke = 'stack_motion_and_poke' in self.config['architecture'] and self.config['architecture']['stack_motion_and_poke']
        self.use_kl_annealing = self.config['training']['kl_annealing'] > 0 if 'kl_annealing' in self.config['training'] else False
        self.kl_weight = 0. if self.use_kl_annealing else self.config['training']['w_kl']
        self.kl_scaling = partial(linear_var,start_it=0, start_val=0, end_val=self.config['training']['w_kl'],
                                  clip_min=0,clip_max=self.config['training']['w_kl'])
        # metrics and discriminators only in train mode

        # metrics
        self.lpips_net = lpips_net()
        self.lpips_metric = LPIPS()
        self.ssim = SSIM_custom()
        self.psnr = PSNR_custom()
        self.FVD = FVD(n_samples=self.config["logging"]["n_samples_fvd"])


        #self.FVD = FVD(n_samples=self.config['logging']['n_samples_fvd'])

        self.vgg_loss = PerceptualLoss() if self.config["training"]["vgg_1"] else VGGLoss()
        self.mf_dt = min(self.config["d_t"]["max_frames"],self.config["data"]["max_frames"])
        self.config["d_t"]['full_seq']=False
        self.disc_t = resnet(config=self.config["d_t"], spatial_size=self.config["data"]["spatial_size"][0],
                             sequence_length=self.mf_dt)
        self.disc_s = PatchDiscriminator(self.config["d_s"])

        self.console_logger = get_logger()

        self.config['architecture'].update({'max_frames': self.config['data']['max_frames']})
        self.config['architecture'].update({'full_seq': self.full_sequence})
        self.enc_motion = resnet18_alternative(dic=self.config['architecture'])

        self.n_layers = self.config['architecture']['n_gru_layers']
        ups = [False] * self.n_layers

        self.rnn = ConvGRU(input_size=self.config['architecture']['z_dim'],
                           hidden_sizes=2 * self.config['architecture']['z_dim'] if self.stack_motion_and_poke else self.config['architecture']['z_dim'],
                           n_layers=self.n_layers,
                           kernel_sizes=3, upsampling=ups)

        self.motion_bias = nn.Parameter(torch.randn(1, config['architecture']['z_dim'],
                                                    config['architecture']['min_spatial_size'],
                                                    config['architecture']['min_spatial_size']), requires_grad=True)

        self.gen = SpadeCondConvDecoder(config['architecture'],stacked_input=self.stack_motion_and_poke)

        n_stages = int(np.log2(self.config["data"]["spatial_size"][0] // self.config["architecture"]["min_spatial_size"]))
        self.poke_enc = ConvEncoder(nf_in=5,nf_max=self.config['architecture']['z_dim'],n_stages=n_stages)

        self.div_scores = []


        self.metrics_dir = os.path.join(self.dirs['generated'],'metrics')
        os.makedirs(self.metrics_dir,exist_ok=True)
        if self.test_mode == 'diversity':
            self.vggm = metric_vgg16()





    def setup(self, stage: str):
        if self.test_mode == 'none':
            assert isinstance(self.logger, WandbLogger)
            self.logger.watch(self,log="all")
        
    def on_train_start(self):
        if self.use_kl_annealing:
            # the actual number of kl_annealing states the number of epochs, within kl_weight is increased until max
            n_increase_it = self.config['training']['kl_annealing'] * self.trainer.num_training_batches
            self.kl_scaling = partial(self.kl_scaling, end_it=n_increase_it)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.use_kl_annealing:
            self.kl_weight = self.kl_scaling(self.global_step)

    def training_step(self, batch, batch_idx, optimizer_idx):

        (opt_g, opt_ds, opt_dt) = self.optimizers()

        # with torch.autograd.detect_anomaly():

        # forward pass
        X = batch['images']
        X_hat, mu, logvar = self(batch)

        log_dict = {}
        if self.current_epoch >= self.config["d_t"]["pretrain"] and self.config["d_t"]["use"]:
            # include also initial frame for smooth transition
            X_fake = torch.cat([X[:,0].unsqueeze(1),X_hat],dim=1)
            sl = X.shape[1]
            offset = int(np.random.choice(np.arange(max(1, sl - self.mf_dt)), 1))
            # offset_fake = int(np.random.choice(np.arange(max(1,seq_len_act-seq_len_temp_disc)), 1))
            X_true = X[:, offset:offset + self.mf_dt].permute(0, 2, 1, 3, 4)
            X_fake = X_fake[:, offset:offset + self.mf_dt].permute(0, 2, 1, 3, 4)
            if self.disc_t.gp_weight > 0.:
                X_true.requires_grad_()
            pred_fake_dt, _ = self.disc_t(X_fake.detach())
            pred_true_dt, _  = self.disc_t(X_true)

            loss_dt_fake = self.disc_t.loss(pred_fake_dt,real=False)
            loss_dt_true = self.disc_t.loss(pred_true_dt,real=True)
            loss_dt = (loss_dt_fake + loss_dt_true) / 2.

            loss_gp_dt  = 0.
            if self.disc_t.gp_weight > 0.:
                loss_gp_dt = self.disc_t.gp2(pred_true_dt,X_true)

            opt_dt.zero_grad()
            loss_dt_all = loss_dt + self.disc_t.gp_weight * loss_gp_dt
            self.manual_backward(loss_dt_all,opt_dt)
            opt_dt.step()

            log_dict.update({"loss_d_dt": loss_dt, f"p_true_dt": torch.sigmoid(pred_true_dt).mean(),
                             "p_fake_dt": torch.sigmoid(pred_fake_dt).mean(),
                             "loss_gp_dt": loss_gp_dt if self.disc_t.gp_weight > 0 else 0})


            self.log("d_t", loss_dt, logger=False, prog_bar=True, on_step=True, on_epoch=False)

        if self.current_epoch >= self.config["d_s"]["pretrain"] and self.config["d_s"]["use"]:
            true_exmpls = np.random.choice(np.arange(X.shape[0] * X.shape[1]), self.config["d_s"]["n_examples"])
            fake_exmpls = np.random.choice(np.arange(X_hat.shape[0] * X_hat.shape[1]), self.config["d_s"]["n_examples"])
            x_true = X.reshape(-1, *X.shape[2:])[true_exmpls]
            x_fake = X_hat.reshape(-1, *X_hat.shape[2:])[fake_exmpls]
            pred_fake_ds, _ = self.disc_s(x_fake.detach())
            pred_true_ds, _ = self.disc_s(x_true)

            loss_ds_fake = self.disc_s.loss(pred_fake_ds, real=False)
            loss_ds_true = self.disc_s.loss(pred_true_ds, real=True)

            opt_ds.zero_grad()
            loss_ds = (loss_ds_fake + loss_ds_true) / 2.
            self.manual_backward(loss_ds, opt_ds)
            opt_ds.step()

            log_dict.update({"loss_d_ds": loss_ds, f"p_true_ds": torch.sigmoid(pred_true_ds).mean(),
                             f"p_fake_ds": torch.sigmoid(pred_fake_ds).mean()})


            self.log("d_s", loss_ds, logger=False, prog_bar=True, on_step=True, on_epoch=False)

        opt_g.zero_grad()

        if self.current_epoch >= self.config["d_s"]["pretrain"] and self.config["d_s"]["use"]:
            pred_fake_ds, _ = self.disc_s(x_fake)
            if self.disc_t.bce_loss:
                loss_gen_ds = self.disc_s.bce(pred_fake_ds, torch.ones_like(pred_fake_ds))
            else:
                loss_gen_ds = -torch.mean(pred_fake_ds)

            # opt_g.zero_grad()
            self.manual_backward(loss_gen_ds,opt_ds,retain_graph=True)

            log_loss_gen_ds = loss_gen_ds.detach()
            log_dict.update({"loss_g_s": log_loss_gen_ds})
            self.log("g_s", log_loss_gen_ds, logger=False, prog_bar=True, on_step=True, on_epoch=False)


        if self.current_epoch >= self.config["d_t"]["pretrain"] and self.config["d_t"]["use"]:
            pred_fake_gen, fmap_fake = self.disc_t(X_fake)
            _, fmap_true = self.disc_t(X_true)

            if self.disc_t.bce_loss:
                loss_gen_dt = self.disc.bce(pred_fake_gen, torch.ones_like(pred_fake_gen))
            else:
                loss_gen_dt = -torch.mean(pred_fake_gen)

            loss_fmap_dt = self.disc_t.fmap_loss(fmap_fake, fmap_true)
            loss_temp = self.config["d_t"]["gen_weight"] * loss_gen_dt + self.config["d_t"]["fmap_weight"] * loss_fmap_dt
            self.manual_backward(loss_temp,opt_g,retain_graph=True)

            log_loss_gen_dt = loss_gen_dt.detach()
            log_dict.update({"loss_g_t": log_loss_gen_dt, "loss_fmap_t": loss_fmap_dt})

            self.log("g_t", log_loss_gen_dt, logger=False, prog_bar=True, on_step=True, on_epoch=False)


        #if not (self.current_epoch > self.config["d_s"]["pretrain"] and self.config["d_s"]["use"]):
        vgg_loss = self.vgg_loss(X[:, 1:].reshape(-1, *X.shape[2:]).contiguous(), X_hat.reshape(-1, *X_hat.shape[2:]).contiguous()).mean()

        l1_loss = torch.mean(torch.abs(X[:, 1:] - X_hat))

        kl_loss = KL(mu, logvar)

        loss = self.config["training"]["w_vgg"] * vgg_loss + self.kl_weight * kl_loss \
               + self.config["training"]["w_l1"] * l1_loss

        self.manual_backward(loss,opt_g)
        opt_g.step()

        log_dict = {"train/" + key: log_dict[key] for key in log_dict}
        self.log_dict(log_dict,prog_bar=False,logger=True,on_step=True,on_epoch=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/l_kl", kl_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/l_rec", l1_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/l_vgg", vgg_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log('train/w_kl',self.kl_weight,logger=False, prog_bar=True, on_epoch=False,on_step=True)
        self.log("iteration", self.global_step)
        self.log("learning rate", opt_g.param_groups[0]["lr"], on_step=True, logger=True)

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            # log video grids
            vid_grid = make_video(X, torch.cat([X[:, 0].unsqueeze(1), X_hat], dim=1),
                                  n_logged=self.config["data"]["batch_size"],
                                  n_max_per_row=int(self.config["data"]["batch_size"] / 2), )
            self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({"Train Video Grid": wandb.Video(vid_grid, caption=f'Training video at it {self.global_step}', fps=3)},
                                       step=self.global_step)

        self.trainer.train_loop.running_loss.append(loss)

    def training_epoch_end(self,outputs):
        self.log("epoch",self.current_epoch)


    def forward_sample(self,batch,n_logged=1,n_samples=1):
        video_samples = []
        with torch.no_grad():
            for n in range(n_samples):
                X_hat_sample, *_ = self(batch, sample_prior=True)
                video_samples.append(X_hat_sample[:n_logged])

        return video_samples

    def validation_step(self,batch,batch_id):

        X = batch["images"]

        with torch.no_grad():
            # forward pass
            X_hat, mu, logvar = self(batch)

            vgg_x = X[:, 1:].reshape(-1, *X.shape[2:]).contiguous()
            vgg_x_hat = X_hat.reshape(-1, *X_hat.shape[2:]).contiguous()
            vgg_loss = self.vgg_loss(vgg_x,vgg_x_hat)

            l1_loss = torch.mean(torch.abs(X[:, 1:] - X_hat))

            log_dict =  {"val/vgg_loss":vgg_loss,"val/rec_loss": l1_loss}
            self.log_dict(log_dict,logger=True,on_epoch=True,on_step=False)

            X_hat_log = X_hat.reshape(-1,*X_hat.shape[2:])
            X_log = X[:,1:].reshape(-1,*X_hat.shape[2:])
            self.log("ssim-val-rec",self.ssim(X_hat_log,X_log).cpu(),on_step=False,on_epoch=True,logger=True)
            self.log("psnr-val-rec", self.psnr(X_hat_log, X_log).cpu(), on_step=False, on_epoch=True, logger=True)
            self.log("lpips-val-rec", self.lpips_metric(self.lpips_net,X_hat_log, X_log).cpu(), on_step=False, on_epoch=True, logger=True)
            #self.FVD.update(X_hat,X)



            if batch_id < self.config["logging"]["n_val_img_batches"]:
                vid_grid = make_video(X, torch.cat([X[:, 0].unsqueeze(1), X_hat], dim=1),
                                      n_logged=self.config["data"]["batch_size"],
                                      n_max_per_row=int(self.config["data"]["batch_size"] / 2))
                self.logger.experiment.history._step = self.global_step
                self.logger.experiment.log({"Val Video Grid Reconstruction": wandb.Video(vid_grid,
                                                                                         caption=f'Validation reconstructions at it {self.global_step}', fps=3)},
                                           step=self.global_step)

                n_logged_vids = self.config['logging']['n_logged_img']

                vid_samples = self.forward_sample(batch,n_logged=n_logged_vids,n_samples=4)
                poke = batch["poke"]
                if isinstance(poke, list):
                    poke_coords = poke[1][:n_logged_vids]
                    poke = poke[0][:n_logged_vids]
                else:
                    poke_coords = None
                    poke = poke[:n_logged_vids]

                poke = poke[0] if isinstance(poke, list) else poke
                flow = batch["flow"][:n_logged_vids]

                vid_sample_grid = make_flow_video_with_samples(X[:n_logged_vids,0],poke,vid_samples,X[:n_logged_vids],flow,
                                                               n_logged=n_logged_vids,
                                                               poke_coords=poke_coords)
                self.logger.experiment.history._step = self.global_step
                self.logger.experiment.log({"Val Video Grid Samples": wandb.Video(vid_sample_grid, caption=f'Validation sample at it {self.global_step}', fps=3)},
                                           step=self.global_step)

            if batch_id <= int(self.config["logging"]["n_samples_fvd"] / X_hat.size(0)):
                X_hat_sample, *_ = self(batch,sample_prior=True)
                self.features_fvd_fake.append(X_hat_sample.cpu().numpy())
                self.features_fvd_true.append(X[:,1:].cpu().numpy())
                self.fvd_features_fake_x0.append(torch.cat([X[:,0].unsqueeze(1),X_hat_sample],dim=1).cpu().numpy())
                self.fvd_features_true_x0.append(X.cpu().numpy())


    def validation_epoch_end(self, outputs) -> None:
        #self.log('FVD-val',self.FVD.compute(),on_step=False,on_epoch=True,logger=True)
    # #     #i3d = I3D(400,"rgb")
    # #
        self.FVD.i3d.eval()
    #
        features_fake = torch.from_numpy(np.concatenate(self.features_fvd_fake,axis=0))
        features_true = torch.from_numpy(np.concatenate(self.features_fvd_true, axis=0))

        fvd_score = calculate_FVD(self.FVD.i3d,features_fake,features_true,
                                  batch_size=self.config["logging"]["bs_i3d"],cuda=True)

        self.log('FVD-val',fvd_score,on_epoch=True,logger=True)

        features_fake = torch.from_numpy(np.concatenate(self.fvd_features_fake_x0, axis=0))
        features_true = torch.from_numpy(np.concatenate(self.fvd_features_true_x0, axis=0))

        fvd_score = calculate_FVD(self.FVD.i3d, features_fake, features_true,
                                  batch_size=self.config["logging"]["bs_i3d"], cuda=True)

        self.log('FVD-val-x0', fvd_score, on_epoch=True, logger=True)

        # self.logger.experiment.history._step = self.global_step
        # self.logger.experiment.log({"FVD-val": fvd_score},step=self.global_step)

        self.features_fvd_fake.clear()
        self.features_fvd_true.clear()
        self.fvd_features_true_x0.clear()
        self.fvd_features_fake_x0.clear()


    def forward(self,batch, sample_prior=False):
        X = batch['images']
        poke = batch['poke']
        if isinstance(poke,list):
            poke_coords = poke[1]
            poke = poke[0]
        else:
            poke_coords = None

        start_frame = X[:,0]
        X_in = X if self.full_sequence else X[:, 1:]

        if sample_prior:
            motion = torch.randn((X_in.size(0),self.config['architecture']['z_dim'],
                                  self.config['architecture']['min_spatial_size'],
                                  self.config['architecture']['min_spatial_size'])).type_as(X_in)
            mu = cov = None
        else:
            motion, mu, cov = self.enc_motion(X_in.transpose(1, 2))

        in_poke_enc = torch.cat([start_frame, poke], dim=1)
        poke_repr, *_ = self.poke_enc(in_poke_enc)
        in_rnn= poke_repr if not self.stack_motion_and_poke else torch.zeros_like(poke_repr)

        # hidden state is initiazed with motion encoding or concatenated motn encoding and poke
        if self.stack_motion_and_poke:
            cat_poke_motion = torch.cat([motion,poke_repr],dim=1)
            hidden = [cat_poke_motion] * self.n_layers
        else:

            hidden = [motion] * self.n_layers
        #ablattmax = X[:,0]

        X_hat = []

        for i in range(X.size(1) - 1):
            hidden = self.rnn(in_rnn, hidden)

            reaction = self.gen([hidden[-1]], start_frame, del_shape=True)

            X_hat.append(reaction)

        X_hat = torch.stack(X_hat, dim=1)

        return X_hat, mu, cov


    def configure_optimizers(
            self,
    ):
        gen_params = [{"params": self.enc_motion.parameters(), "name": "motion_encoder"},
                      {"params": self.poke_enc.parameters(), "name": "poke_encoder"},
                      {"params": self.rnn.parameters(), "name": "rnn"},
                     {"params": self.gen.parameters(), "name": "decoder"}
                     ]
        dt_params = [{"params": self.disc_t.parameters(),"name":"disc_temp"}]
        ds_params = [{"params": self.disc_s.parameters(),"name":"disc_spatial"}]

        opt_g = Adam(gen_params,lr=self.config["training"]["lr"],betas=(0.5,0.9),weight_decay=self.config["training"]["weight_decay"])
        opt_dt = Adam(dt_params,lr=self.config["training"]["lr"],betas=(0.5,0.9),weight_decay=self.config["training"]["weight_decay"])
        opt_ds = Adam(ds_params, lr=self.config["training"]["lr"], betas=(0.5, 0.9), weight_decay=self.config["training"]["weight_decay"])

        sched_g =  torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.config["training"]['gamma'])
        sched_dt = torch.optim.lr_scheduler.ExponentialLR(opt_dt, gamma=self.config["training"]["gamma"])
        sched_ds = torch.optim.lr_scheduler.ExponentialLR(opt_ds, gamma=self.config["training"]["gamma"])

        return [opt_g,opt_ds,opt_dt],[sched_g, sched_ds, sched_dt]


    def _test_step_fvd(self,batch):
        self.eval()


        with torch.no_grad():
            X = batch['images']

            sample, *_ = self(batch,sample_prior=True)

            out_real = ((X + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
            out_gen = torch.cat([X[:, 0].unsqueeze(1), sample], dim=1)
            out_gen = ((out_gen + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)

            self.fvd_features_fake_x0.append(out_gen)
            self.fvd_features_true_x0.append(out_real)

    def _test_step_diversity(self, batch, batch_id):
        self.eval()
        n_samples = self.config['testing']['n_samples_per_data_point']
        samples = []

        # start_frame = batch['images'][:,0][:,None]
        for n in tqdm(range(n_samples),desc=f'Generating {n_samples} for current data point...'):
            sample, *_ = self(batch, sample_prior=True)
            samples.append(sample)

        samples = torch.stack(samples, dim=1)
        # self.test_aggr.append(samples)

        return samples

    def test_step_end(self, out_step):
        return out_step

    def _generate_samples(self,batch):
        self.eval()
        start_id = batch['sample_ids'][:,0].cpu().numpy()
        n_test_samples = self.config['testing']['n_samples_per_data_point']
        # start_frame = batch['images'][:, 0][:, None]
        samples = []

        for n in range(n_test_samples):
            sample, *_ = self.forward_sample(batch,self.n_test_samples)
            samples.append(sample)

        # change dimensions to obtain samppermuteles in right order
        samples = list(torch.stack(samples,dim=1))

        x_0s = batch["images"][:, 0]
        X_tgts = batch["images"]
        pokes = batch["poke"]
        if isinstance(pokes, list):
            poke_coords = pokes[1]
            pokes = pokes[0]
        else:
            poke_coords = None


        for sid,x_0,poke,X_tgt,s,poke_coord in zip(start_id,x_0s,pokes,X_tgts,samples,poke_coords):

            x_0 = x_0[None]
            poke = poke[None]
            X_tgt = X_tgt[None]
            poke_coord = poke_coord[None]
            samples_list, samples_grid = make_samples_and_samplegrid(x_0,poke,X_tgt,s,
                                                                     poke_coords=poke_coord)

            if self.config['general']['last_ckpt']:
                samples_dir = 'samples_last_ckpt'
            else:
                samples_dir = 'samples_best_fvd'

            savedir = os.path.join(self.dirs['generated'],samples_dir,f'sid_{sid}')
            os.makedirs(savedir,exist_ok=True)

            save_video(samples_grid,os.path.join(savedir,'overview.mp4'),fps=3)

            for i,sample in enumerate(samples_list):
                if i==0:
                    savepath = os.path.join(savedir, f'groundtruth.mp4')
                else:
                    savepath = os.path.join(savedir,f'sample_{i}.mp4')
                save_video(sample,savepath,fps=3)

    def test_step(self, batch, batch_id):
        self.eval()
        with torch.no_grad():
            if self.test_mode == 'fvd':
                self._test_step_fvd(batch)
            elif self.test_mode == 'metrics':
                # self._test_step_metrics(batch,batch_id)
                raise NotImplementedError()
            elif self.test_mode == 'samples':
                self._generate_samples(batch)
            elif self.test_mode == 'kps_acc':
                raise NotImplementedError()
            elif self.test_mode == 'diversity':
                samples = self._test_step_diversity(batch,batch_id)
                return samples
            else:
                raise ValueError(f'The specified test_mode is "{self.test_mode}", which is invalid...')



    def test_epoch_end(self, outputs):

        self.console_logger.info(f'******************* TEST SUMMARY on {self.trainer.datamodule.dset_val.__class__.__name__} FOR {self.config["testing"]["n_samples_per_data_point"]} SAMPLES *******************')

        if self.test_mode == 'fvd':
            savedir = os.path.join(self.dirs['generated'],'samples_fvd')
            savedir_vid_samples = os.path.join(self.dirs['generated'],'vid_examples')

            os.makedirs(savedir, exist_ok=True)
            os.makedirs(savedir_vid_samples, exist_ok=True)

            real_samples = np.stack(self.fvd_features_true_x0, axis=0)
            fake_samples = np.stack(self.fvd_features_fake_x0, axis=0)

            self.console_logger.info(f"Generating example videos")
            for i, (r, f) in enumerate(zip(real_samples, fake_samples)):
                savename = os.path.join(savedir_vid_samples, f"sample{i}.mp4")
                r = np.concatenate([v for v in r], axis=2)
                f = np.concatenate([v for v in f], axis=2)
                all = np.concatenate([r, f], axis=1)

                save_video(all, savename)

                if i >= 4:
                    break

            self.console_logger.info(f"Saving samples to {savedir}")
            np.save(os.path.join(savedir, "real_samples.npy"), real_samples)
            np.save(os.path.join(savedir, "fake_samples.npy"), fake_samples)

            self.console_logger.info(f'Finish generation of vid samples.')


        elif self.test_mode == 'diversity':
            n_pokes = self.trainer.datamodule.dset_val.config['n_pokes']
            samples = outputs
            exmpls = torch.cat(samples,dim=0)
            div_score = compute_div_score(exmpls,self.vggm,device=self.device)
            div_score_mse = compute_div_score_mse(exmpls,device=self.device)
            div_score_lpips = compute_div_score_lpips(exmpls,device=self.device)
            self.div_scores.append(div_score)
            exmpls = exmpls.cpu().numpy()
            savepath = os.path.join(self.dirs['generated'],'diversity')
            os.makedirs(savepath, exist_ok=True)

            np.save(os.path.join(savepath,f'samples_diversity_{n_pokes}_pokes.npy'),exmpls)

            text =f'Similarity measure_vgg: {div_score}; similarity measure mse: {div_score_mse}; similarity measure lpips: {div_score_lpips}\n'
            self.console_logger.info(text)
            #self.console_logger.info(f'Average cosine distance in vgg features space for {n_pokes} pokes: {div_score}')
            divscore_path = os.path.join(self.metrics_dir,f'divscore.txt')
            with open(divscore_path,'a+') as f:
                f.writelines(text)

            return div_score