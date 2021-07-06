import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
import wandb
import numpy as np
import os

from models.modules.motion_models.motion_encoder import resnet18,resnet18_alternative
from models.modules.motion_models.motion_generator import Generator
from models.modules.discriminators.patchgan import PatchDiscriminator
from models.modules.discriminators.patchgan_3d import resnet
from models.modules.autoencoders.LPIPS import LPIPS as PerceptualLoss
from models.modules.autoencoders.fully_conv_models import ConvDecoder, ConvEncoder, SpadeCondConvDecoder
from models.modules.autoencoders.baseline_fc_models import BaselineFCGenerator
from models.modules.motion_models.rnn import ConvGRU
from models.modules.autoencoders.util import ResBlock
from lpips import LPIPS as lpips_net
from utils.metrics import LPIPS, SSIM_custom, PSNR_custom, FVD,I3D,calculate_FVD
from utils.losses import KL, VGGLoss
from utils.logging import make_video
from utils.general import get_logger


class MotionModel(pl.LightningModule):

    def __init__(self,config,train=True):
        super().__init__()
        self.config = config
        #self.disc_start = self.config["training"]['pretrain']
        self.config["architecture"].update({"img_size":self.config["data"]["spatial_size"][0]})

        # # features for fvd calculattion
        self.features_fvd_true = []
        self.features_fvd_fake = []
        self.fvd_features_true_x0 = []
        self.fvd_features_fake_x0 = []

        self.full_sequence = 'full_sequence' in self.config['training'] and self.config['training']['full_sequence']
        #self.i3d = I3D(400, 'rgb')
        # model_path = '/export/data/ablattma/i3d_kinetics_rgb.pth'
        # if 'DATAPATH' in os.environ:
        #     model_path = os.path.join(os.environ['DATAPATH'], model_path[1:])
        # # model_path = '/export/home/ablattma/tools/kinetics-i3d-Pytorch/data/pytorch_checkpoints/rgb_imagenet.pkl'
        # state_dict = torch.load(model_path, map_location="cpu")
        # #state_dict = {'.'.join(key.split('.')[1:]):state_dict[key] for key in state_dict}
        # self.i3d.load_state_dict(state_dict)

        # metrics and discriminators only in train mode

        if train:
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

        if issubclass(self.__class__,MotionModel):
            return

        # generator
        self.config['architecture'].update({'max_frames':self.config['data']['max_frames']})
        self.config['architecture'].update({'full_seq': self.full_sequence})
        if self.full_sequence:
            self.console_logger.info('Training motion encoder on full sequence.')

        self.enc_motion = resnet18_alternative(dic=self.config["architecture"])
        self.gen = Generator(self.config["architecture"])







    def setup(self, stage: str):
        assert isinstance(self.logger, WandbLogger)
        self.logger.watch(self,log="all")

    def forward(self,X):

        X_in = X if self.full_sequence else X[:,1:]

        motion, mu, cov = self.enc(X_in.transpose(1,2))
        X_hat = self.gen(X[:,0],motion)

        return X_hat, mu, cov

    def train_disc(self, x_in_true,x_in_fake,disc,opt):

        postfix = "temp" if x_in_fake.dim() == 5 else "static"

        disc.train()

        # if self.parallel:
        #     x_in_true = x_in_true.cuda(self.devices[0])
        #     x_in_fake = x_in_fake.cuda(self.devices[0])
        # set gradient to zero
        opt.zero_grad()

        # real examples
        x_in_true.requires_grad_()

        pred_true, _ = disc(x_in_true)
        loss_real = disc.loss(pred_true, real=True)
        if disc.gp_weight > 0:
            loss_real.backward(retain_graph=True)
            # gradient penalty
            loss_gp = disc.gp(pred_true, x_in_true).mean()
            gp_weighted = disc.gp_weight * loss_gp
            self.manual_backward(gp_weighted,opt)
        else:
            self.manual_backward(loss_real,opt)

        # fake examples
        pred_fake, _ = disc(x_in_fake.detach())
        loss_fake = disc.loss(pred_fake, real=False)
        self.manual_backward(loss_fake,opt)

        # optmize parameters
        opt.step()

        loss_disc = ((loss_real + loss_fake) / 2.).item()
        out_dict = {f"loss_d_{postfix}": loss_disc, f"p_true_{postfix}": torch.sigmoid(pred_true).mean().item(), f"p_fake_{postfix}": torch.sigmoid(pred_fake).mean().item(),
                    f"loss_gp_{postfix}": loss_gp.item() if disc.gp_weight > 0 else 0}

        # train generator
        pred_fake, fmap_fake = disc(x_in_fake)
        _, fmap_true = disc(x_in_true)
        if disc.bce_loss:
            loss_gen =disc.bce(pred_fake, torch.ones_like(pred_fake))
        else:
            loss_gen = -torch.mean(pred_fake)

        loss_fmap = disc.fmap_loss(fmap_fake, fmap_true)
        # if self.parallel:
        #     loss_fmap = loss_fmap.cuda(self.devices[0])
        #     loss_gen = loss_gen.cuda(self.devices[0])

        return out_dict, loss_gen, loss_fmap

    def training_step(self, batch, batch_idx, optimizer_idx):

        (opt_g, opt_ds, opt_dt) = self.optimizers()

        X = batch["images"]

        # with torch.autograd.detect_anomaly():

        # forward pass
        X_hat, mu, logvar = self(X)




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
        if self.enc_motion.be_determinstic:
            kl_loss = 0.
        else:
            kl_loss = KL(mu, logvar)

        loss = self.config["training"]["w_vgg"] * vgg_loss + self.config["training"]["w_kl"] * kl_loss \
               + self.config["training"]["w_l1"] * l1_loss


        self.manual_backward(loss,opt_g)
        opt_g.step()

        log_dict = {"train/" + key: log_dict[key] for key in log_dict}
        self.log_dict(log_dict,prog_bar=False,logger=True,on_step=True,on_epoch=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/l_kl", kl_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/l_rec", l1_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/l_vgg", vgg_loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

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


    def validation_step(self,batch,batch_id):

        X = batch["images"]

        with torch.no_grad():
            # forward pass
            X_hat, mu, logvar = self(X)

            vgg_x = X[:, 1:].reshape(-1, *X.shape[2:]).contiguous()
            vgg_x_hat = X_hat.reshape(-1, *X_hat.shape[2:]).contiguous()
            vgg_loss = self.vgg_loss(vgg_x,vgg_x_hat)

            l1_loss = torch.mean(torch.abs(X[:, 1:] - X_hat))

            log_dict =  {"val/vgg_loss":vgg_loss,"val/rec_loss": l1_loss}
            self.log_dict(log_dict,logger=True,on_epoch=True,on_step=False)

            X_hat_log = X_hat.reshape(-1,*X_hat.shape[2:])
            X_log = X[:,1:].reshape(-1,*X_hat.shape[2:])
            self.log("ssim-val",self.ssim(X_hat_log,X_log).cpu(),on_step=False,on_epoch=True,logger=True)
            self.log("psnr-val", self.psnr(X_hat_log, X_log).cpu(), on_step=False, on_epoch=True, logger=True)
            self.log("lpips-val", self.lpips_metric(self.lpips_net,X_hat_log, X_log).cpu(), on_step=False, on_epoch=True, logger=True)
            #self.FVD.update(X_hat,X)

            if batch_id < self.config["logging"]["n_val_img_batches"]:
                vid_grid = make_video(X, torch.cat([X[:, 0].unsqueeze(1), X_hat], dim=1),
                                      n_logged=self.config["data"]["batch_size"],
                                      n_max_per_row=int(self.config["data"]["batch_size"] / 2))
                self.logger.experiment.history._step = self.global_step
                self.logger.experiment.log({"Val Video Grid": wandb.Video(vid_grid, caption=f'Validation video at it {self.global_step}', fps=3)},
                                           step=self.global_step)


            if batch_id <= int(self.config["logging"]["n_samples_fvd"] / X_hat.size(0)):
                self.features_fvd_fake.append(X_hat.cpu().numpy())
                self.features_fvd_true.append(X[:,1:].cpu().numpy())
                self.fvd_features_fake_x0.append(torch.cat([X[:,0].unsqueeze(1),X_hat],dim=1).cpu().numpy())
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


    def configure_optimizers(
            self,
    ):
        gen_params = [{"params": self.enc.parameters(), "name": "encoder"},
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


class RNNMotionModel(MotionModel):

    def __init__(self,config,train=True):
        super().__init__(config,train)

        n_stages = int(np.log2(self.config["data"]["spatial_size"][0] // self.config["architecture"]["min_spatial_size"]))


        self.config['architecture'].update({'max_frames': self.config['data']['max_frames']})
        self.enc_motion = resnet18_alternative(dic=self.config['architecture'])
        self.enc_static = ConvEncoder(nf_in=3,nf_max=self.config['architecture']['z_dim'],
                                      n_stages=n_stages)
        self.n_layers = self.config['architecture']['n_gru_layers']
        ups = [False] * self.n_layers

        self.rnn = ConvGRU(input_size=self.config['architecture']['z_dim'],
                           hidden_sizes=self.config['architecture']['z_dim'],
                           n_layers=self.n_layers,
                           kernel_sizes=3,upsampling=ups)
        self.post_hidden = ResBlock(self.config['architecture']['z_dim'],
                                    self.config['architecture']['z_dim'],norm='group',snorm=True)

        decoder_channels = self.config['architecture']['dec_channels']
        self.gen = ConvDecoder(nf_in=self.config['architecture']['z_dim'],in_channels=decoder_channels)




    def forward(self,X):

        scene = self.enc_static(X[:,0])[0]
        X_in = X if self.full_sequence else X[:, 1:]
        motion, mu, cov = self.enc_motion(X_in.transpose(1,2))

        # hidden state is initiazed with motion encoding
        hidden = [motion] * self.n_layers
        x = scene

        X_hat = []
        for i in range(X.size(1)-1):
            hidden = self.rnn(x,hidden)
            x = self.post_hidden(hidden[-1])
            reaction = self.gen([x],del_shape=True)

            X_hat.append(reaction)

        X_hat = torch.stack(X_hat,dim=1)

        return X_hat, mu, cov


    def configure_optimizers(
            self,
    ):
        gen_params = [{"params": self.enc_motion.parameters(), "name": "motion_encoder"},
                      {"params": self.enc_static.parameters(), "name": "image_encoder"},
                      {"params": self.post_hidden.parameters(), "name": "post_hidden"},
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


class SpadeCondMotionModel(MotionModel):

    def __init__(self,config,dirs,train=True):
        super().__init__(config,train)

        self.dirs = dirs
        self.use_motion_bias = 'motion_bias' in self.config['architecture'] and self.config['architecture']['motion_bias']

        n_stages = int(
            np.log2(self.config["data"]["spatial_size"][0] // self.config["architecture"]["min_spatial_size"]))

        self.config['architecture'].update({'max_frames': self.config['data']['max_frames']})
        self.config['architecture'].update({'full_seq': self.full_sequence})
        self.enc_motion = resnet18_alternative(dic=self.config['architecture'])

        self.n_layers = self.config['architecture']['n_gru_layers']
        ups = [False] * self.n_layers

        self.rnn = ConvGRU(input_size=self.config['architecture']['z_dim'],
                           hidden_sizes=self.config['architecture']['z_dim'],
                           n_layers=self.n_layers,
                           kernel_sizes=3, upsampling=ups)
        if self.use_motion_bias:
            self.motion_bias  = nn.Parameter(torch.randn(1,config['architecture']['z_dim'],
                                                     config['architecture']['min_spatial_size'],
                                                     config['architecture']['min_spatial_size']),requires_grad=True)

        self.gen = SpadeCondConvDecoder(config['architecture'])

    def forward(self,X):
        start_frame = X[:,0]
        X_in = X if self.full_sequence else X[:, 1:]

        motion, mu, cov = self.enc_motion(X_in.transpose(1, 2))

        # hidden state is initiazed with motion encoding
        hidden = [motion] * self.n_layers
        #x = X[:,0]

        X_hat = []
        if self.use_motion_bias:
            in_rnn = torch.cat([self.motion_bias]*start_frame.size(0),dim=0)
        else:
            in_rnn = motion
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
                      {"params": self.rnn.parameters(), "name": "rnn"},
                     {"params": self.gen.parameters(), "name": "decoder"}
                     ]
        if self.use_motion_bias:
            gen_params.append({"params": self.motion_bias   , "name": "motion_bias"})

        dt_params = [{"params": self.disc_t.parameters(),"name":"disc_temp"}]
        ds_params = [{"params": self.disc_s.parameters(),"name":"disc_spatial"}]

        opt_g = Adam(gen_params,lr=self.config["training"]["lr"],betas=(0.5,0.9),weight_decay=self.config["training"]["weight_decay"])
        opt_dt = Adam(dt_params,lr=self.config["training"]["lr"],betas=(0.5,0.9),weight_decay=self.config["training"]["weight_decay"])
        opt_ds = Adam(ds_params, lr=self.config["training"]["lr"], betas=(0.5, 0.9), weight_decay=self.config["training"]["weight_decay"])

        sched_g =  torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.config["training"]['gamma'])
        sched_dt = torch.optim.lr_scheduler.ExponentialLR(opt_dt, gamma=self.config["training"]["gamma"])
        sched_ds = torch.optim.lr_scheduler.ExponentialLR(opt_ds, gamma=self.config["training"]["gamma"])

        return [opt_g,opt_ds,opt_dt],[sched_g, sched_ds, sched_dt]


class FCBaseline(MotionModel):

    def __init__(self,config,dirs,train=True):
        super().__init__(config,train)

        self.dirs = dirs
        self.enc_motion = resnet18(dic=self.config['architecture'])

        latent_dim = self.config['architecture']['z_dim']
        self.motion_bias = nn.Parameter(torch.randn((1,latent_dim),requires_grad=True))
        self.n_layers = self.config['architecture']['n_gru_layers']
        self.rnn = nn.GRU(input_size=latent_dim,hidden_size=latent_dim,
                          num_layers=self.n_layers,batch_first=True)

        self.gen = BaselineFCGenerator(self.config['architecture'])

    def forward(self,X):
        start_frame = X[:,0]
        X_in = X if self.full_sequence else X[:, 1:]

        motion, mu, cov = self.enc_motion(X_in.transpose(1, 2))

        # hidden state is initiazed with motion encoding
        hidden = torch.stack([motion] * self.n_layers,dim=0)
        #x = X[:,0]

        X_hat = []

        in_rnn = torch.cat([self.motion_bias]*start_frame.size(0),dim=0)[:,None]

        for i in range(X.size(1) - 1):
            out,hidden = self.rnn(in_rnn, hidden)

            reaction = self.gen([out.squeeze(1)], start_frame, del_shape=True)

            X_hat.append(reaction)

        X_hat = torch.stack(X_hat, dim=1)

        # return mu and conv as 4d tensor to be able to use loss framework from main model
        return X_hat, mu[...,None,None], cov[...,None,None]

    def configure_optimizers(
            self,
    ):
        gen_params = [{"params": self.enc_motion.parameters(), "name": "motion_encoder"},
                      {"params": self.rnn.parameters(), "name": "rnn"},
                      {"params": self.motion_bias, "name": "motion_bias"},
                     {"params": self.gen.parameters(), "name": "decoder"}]

        dt_params = [{"params": self.disc_t.parameters(),"name":"disc_temp"}]
        ds_params = [{"params": self.disc_s.parameters(),"name":"disc_spatial"}]

        opt_g = Adam(gen_params,lr=self.config["training"]["lr"],betas=(0.5,0.9),weight_decay=self.config["training"]["weight_decay"])
        opt_dt = Adam(dt_params,lr=self.config["training"]["lr"],betas=(0.5,0.9),weight_decay=self.config["training"]["weight_decay"])
        opt_ds = Adam(ds_params, lr=self.config["training"]["lr"], betas=(0.5, 0.9), weight_decay=self.config["training"]["weight_decay"])

        sched_g =  torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.config["training"]['gamma'])
        sched_dt = torch.optim.lr_scheduler.ExponentialLR(opt_dt, gamma=self.config["training"]["gamma"])
        sched_ds = torch.optim.lr_scheduler.ExponentialLR(opt_ds, gamma=self.config["training"]["gamma"])

        return [opt_g,opt_ds,opt_dt],[sched_g, sched_ds, sched_dt]