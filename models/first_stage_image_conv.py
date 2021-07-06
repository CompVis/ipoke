import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam, lr_scheduler
import wandb
import numpy as np
import logging

from models.modules.autoencoders.fully_conv_models import ConvDecoder,ConvEncoder
from models.modules.autoencoders.LPIPS import LPIPS as PerceptualLoss
from models.modules.discriminators.disc_utils import calculate_adaptive_weight, adopt_weight, hinge_d_loss
from models.modules.discriminators.patchgan import define_D
from utils.metrics import LPIPS, SSIM_custom, PSNR_custom, FIDInceptionModel, compute_fid
from lpips import LPIPS as lpips_net
from utils.logging import batches2image_grid
from utils.losses import kl_conv
from utils.general import get_logger


class ConvAEModel(pl.LightningModule):

    def __init__(self,config):
        super().__init__()
        #self.automatic_optimization=False
        self.config = config
        self.be_deterministic = self.config["architecture"]['deterministic']
        self.kl_weight = self.config["training"]['w_kl']
        self.register_buffer("disc_factor",torch.tensor(1.),persistent=True)
        self.register_buffer("disc_weight",torch.tensor(1.),persistent=True)
        self.register_buffer("perc_weight",torch.tensor(1.),persistent=True)
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)
        self.disc_start = self.config["training"]['pretrain']
        self.n_logged_imgs = self.config["logging"]["n_log_images"]
        self.forward_sample = self.config["training"]["forward_sample"] if "forward_sample" in self.config["training"] else False

        self.vgg_loss = PerceptualLoss()

        # (v)ae
        n_stages = int(np.log2(self.config["data"]["spatial_size"][0] // self.config["architecture"]["min_spatial_size"]))
        self.encoder = ConvEncoder(nf_in=self.config["architecture"]["nf_in"],nf_max=self.config["architecture"]["nf_max"],
                                   n_stages=n_stages,variational=not self.be_deterministic)
        decoder_channels = [self.config["architecture"]["nf_max"]] + self.encoder.depths
        self.decoder = ConvDecoder(self.config["architecture"]["nf_max"],decoder_channels)


        # discriminator
        self.disc = define_D(3, 64, netD='basic',gp_weight=self.config["training"]["gp_weight"])

        # metrics
        self.ssim = SSIM_custom()
        self.psnr = PSNR_custom()
        self.lpips_net = lpips_net()
        for param in self.lpips_net.parameters():
            param.requires_grad = False

        self.lpips_metric = LPIPS()
        self.inception_model = FIDInceptionModel(normalize_range=True)

        self.fid_features_real = []
        self.fid_features_fake = []

        # self.logger = get_logger()

        self.n_it_fid = int(np.ceil(2000 / self.config["data"]["batch_size"]))

    def setup(self, stage: str):
        assert isinstance(self.logger, WandbLogger)
        self.logger.watch(self,log="all")


    def forward(self,x):
        """

        :param x: The Image to b reconstructed (will be later the video to be reconstructed)
        :return:
        """
        p_s, mu, log_sigma = self.encoder(x)

        in_dec = p_s if self.forward_sample or self.be_deterministic else mu
        img = self.decoder([in_dec],del_shape=False)

        return img, mu, log_sigma

    def train_disc(self,x_in_true,x_in_fake,opt):
        if self.current_epoch < self.disc_start:
            return {}, torch.zeros_like(x_in_fake).mean()
        self.disc.train()

        opt.zero_grad()

        x_in_true.requires_grad_()

        pred_true = self.disc(x_in_true)
        loss_real = self.disc.loss(pred_true, real=True)
        if self.disc.gp_weight > 0:
            loss_real.backward(retain_graph=True)
            # gradient penalty
            loss_gp = self.disc.gp(pred_true, x_in_true).mean()
            gp_weighted = self.disc.gp_weight * loss_gp
            self.manual_backward(gp_weighted, opt)
        else:
            self.manual_backward(loss_real, opt)

        # fake examples
        pred_fake = self.disc(x_in_fake.detach())
        loss_fake = self.disc.loss(pred_fake, real=False)
        self.manual_backward(loss_fake, opt)

        # optmize parameters
        opt.step()

        loss_disc = ((loss_real + loss_fake) / 2.).item()
        out_dict = {f"train/d_loss": loss_disc, f"train/p_true": torch.sigmoid(pred_true).mean().item(), f"train/p_fake": torch.sigmoid(pred_fake).mean().item(),
                    f"train/gp_loss": loss_gp.item() if self.disc.gp_weight > 0 else 0}

        # train generator
        pred_fake= self.disc(x_in_fake)
        if self.disc.bce_loss:
            loss_gen = self.disc.bce(pred_fake, torch.ones_like(pred_fake))
        else:
            loss_gen = -torch.mean(pred_fake)

        #loss_fmap = self.disc.fmap_loss(fmap_fake, fmap_true)
        # if self.parallel:
        #     loss_fmap = loss_fmap.cuda(self.devices[0])
        #     loss_gen = loss_gen.cuda(self.devices[0])

        return out_dict, loss_gen #, loss_fmap


    def training_step(self,batch, batch_idx,optimizer_idx):
        x = batch["images"][:,-1]

        (opt_g, opt_d) = self.optimizers()

        rec, mu, log_sigma = self(x)
        rec_loss = torch.abs(x.contiguous() - rec.contiguous())

        p_loss = self.vgg_loss(x.contiguous(), rec.contiguous())
        # equal weighting of l1 and perceptual loss
        rec_loss = rec_loss +  self.perc_weight * p_loss
        if self.be_deterministic:
            kl_loss = 0.
        else:
            kl_loss = kl_conv(mu,log_sigma)

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        d_dict, g_loss = self.train_disc(x,rec,opt_d)

        # # generator update
        # logits_fake = self.discriminator(rec)
        # g_loss = -torch.mean(logits_fake)


        d_weight = calculate_adaptive_weight(nll_loss, g_loss, self.disc_weight,
                                             last_layer=list(self.decoder.parameters())[-1])\
            if self.current_epoch >= self.disc_start else 0

        disc_factor = adopt_weight(self.disc_factor, self.current_epoch, threshold=self.disc_start)
        loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss


        opt_g.zero_grad()
        self.manual_backward(loss,opt_g)
        opt_g.step()

        mean_rec_loss = rec_loss.mean()
        loss_dict = {"train/loss": loss, "train/kl_loss": kl_loss, "train/logvar": self.logvar.detach(), "train/nll_loss":nll_loss,
                      "train/rec_loss": mean_rec_loss,"train/d_weight":d_weight, "train/disc_factor": disc_factor,"train/g_loss": g_loss,}
        loss_dict.update(d_dict)

        self.log_dict(loss_dict,logger=True,on_epoch=True,on_step=True)
        #self.logger.experiment.log({k: loss_dict[k].item() if isinstance(loss_dict[k],torch.Tensor) else loss_dict[k] for k in loss_dict})
        self.log("global step", self.global_step)
        self.log("learning rate",opt_g.param_groups[0]["lr"],on_step=True, logger=True)

        #self.log_dict(loss_dict, prog_bar=True, on_step=True, logger=False)

        self.log("overall_loss",loss,prog_bar=True,logger=False)
        self.log("d_loss",d_dict["train/d_loss"] if "train/d_loss" in d_dict else 0,prog_bar=True,logger=False)
        self.log("kl_loss",kl_loss,prog_bar=True,logger=False)
        self.log("nll_loss",nll_loss,prog_bar=True,logger=False)
        self.log("g_loss",g_loss,prog_bar=True,logger=False)
        self.log("logvar",self.logvar.detach(),prog_bar=True,logger=False)
        self.log("rec_loss",mean_rec_loss,prog_bar=True,logger=False)

        loss_dict.update({"img_real-train": x, "img_fake-train": rec})

        return loss_dict, batch_idx

    def training_step_end(self, outputs):

        # for convenience, in case ditributed training is used
        loss_dict = outputs[0]
        x  =loss_dict["img_real-train"]
        rec = loss_dict["img_fake-train"]
        #

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            imgs = [x, rec]
            captions = ["Targets", "Predictions"]
            train_grid = batches2image_grid(imgs, captions)
            self.logger.experiment.log({f"Train Batch": wandb.Image(train_grid,
                                                                    caption=f"Training Images @ it #{self.global_step}")},step=self.global_step)

    def training_epoch_end(self, outputs):
        self.log("epoch",self.current_epoch)

    def validation_step(self, batch, batch_id):
        # dataloader yields pair of images; here we consider only the last element of this pair
        with torch.no_grad():
            x = batch["images"][:,-1]

            rec, mu, log_sigma = self(x)
            rec_loss = torch.abs(x.contiguous() - rec.contiguous())

            p_loss = self.vgg_loss(x.contiguous(), rec.contiguous())
            # equal weighting of l1 and perceptual loss
            rec_loss = rec_loss + self.perc_weight * p_loss


            kl_loss = 0. if self.be_deterministic else kl_conv(mu,log_sigma)

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

            loss = nll_loss + self.kl_weight * kl_loss

            log_dict = {"val/loss": loss, "val/kl_loss": kl_loss, "val/logvar": self.logvar.detach(),
                         "val/nll_loss": nll_loss}

            self.log_dict(log_dict, logger=True, prog_bar=False,on_epoch=True)

            log_dict.update({"img_real-val": x, "img_fake-val": rec})

            if batch_id < self.n_it_fid:
                self.fid_features_real.append(self.inception_model(x).cpu().numpy())
                self.fid_features_fake.append(self.inception_model(rec).cpu().numpy())

        return log_dict, batch_id

    def validation_step_end(self,val_out):
        log_dict = val_out[0]
        batch_id = val_out[1]
        x = log_dict["img_real-val"]
        rec = log_dict["img_fake-val"]

        # # log train metrics
        with torch.no_grad():
            self.log("ssim-val", self.ssim(rec, x), on_step=False, on_epoch=True)
            self.log("psnr-val", self.psnr(rec, x), on_step=False, on_epoch=True)
            # self.log("val/lpips", lpips, on_step=False, on_epoch=True)
            self.log("lpips-val", self.lpips_metric(self.lpips_net,rec,x), on_step=False, on_epoch=True,logger=True)

        if batch_id < self.config["logging"]["n_val_img_batches"]:
            imgs = [x[:self.n_logged_imgs],rec[:self.n_logged_imgs]]
            captions = ["Targets", "Predictions"]
            val_grid = batches2image_grid(imgs,captions)
            self.logger.experiment.log({f"Validation Batch #{batch_id}" : wandb.Image(val_grid,
                                                                                      caption=f"Validation Images @ it {self.global_step}")},step=self.global_step)

    def validation_epoch_end(self,outputs):
        fid = compute_fid(self.fid_features_real,self.fid_features_fake)
        #self.logger.info(f'FID after validation @epoch {self.current_epoch}: {fid}')
        self.logger.experiment.history._step = self.global_step
        self.logger.experiment.log({"fid-val": fid},step=self.global_step)
        self.fid_features_fake.clear()
        self.fid_features_real.clear()

    def configure_optimizers(self):
        # optimizers
        ae_params = [{"params": self.encoder.parameters(), "name": "encoder"},
                     {"params": self.logvar, "name": "logvar"},
                      {"params": self.decoder.parameters(), "name": "decoder"}
            ]
        lr = self.config["training"]["lr"]


        opt_g = Adam(ae_params, lr = lr,weight_decay=self.config["training"]["weight_decay"])
        opt_d = Adam(self.disc. parameters(),lr=self.config["training"]["lr"],
                     weight_decay=self.config["training"]["weight_decay"])

        # schedulers
        sched_g = lr_scheduler.ReduceLROnPlateau(opt_g,mode="min",factor=.5,patience=0,min_lr=1e-8,
                                                             threshold=0.001, threshold_mode='rel')
        sched_d = lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=.1, patience=0, min_lr=1e-8,
                                                 threshold=0.001, threshold_mode='rel')

        return [opt_g,opt_d], [{'scheduler':sched_g,'monitor':"val/loss","interval":1,'reduce_on_plateau':True,'strict':True},
                               {'scheduler':sched_d,'monitor':"val/loss","interval":1,'reduce_on_plateau':True,'strict':True}]
        # return ({'optimizer': opt_g,'lr_scheduler':sched_g,'monitor':"loss-val","interval":1,'reduce_on_plateau':True,'strict':True},
                # {'optimizer': opt_d,'lr_scheduler':sched_d,'monitor':"loss-val","interval":1,'reduce_on_plateau':True,'strict':True})
