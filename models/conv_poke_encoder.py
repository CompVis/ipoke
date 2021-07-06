import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.regression import SSIM, PSNR
from torch.optim import Adam, lr_scheduler
import wandb

from models.modules.autoencoders.fully_conv_models import FirstStageWrapper
from models.modules.autoencoders.LPIPS import LPIPS as PerceptualLoss
from utils.metrics import LPIPS
from lpips import LPIPS as lpips_net
from utils.logging import batches2flow_grid


class ConvPokeAE(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        # self.automatic_optimization=False
        self.config = config
        self.be_deterministic = self.config["architecture"]['deterministic']
        self.kl_weight = self.config["training"]['w_kl']
        self.register_buffer("disc_factor", torch.tensor(1.), persistent=True)
        self.register_buffer("disc_weight", torch.tensor(1.), persistent=True)
        self.register_buffer("perc_weight", torch.tensor(1.), persistent=True)
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)
        self.n_logged_imgs = self.config["logging"]["n_log_images"]
        self.flow_ae = "flow_ae" in self.config["architecture"] and self.config["architecture"]["flow_ae"]
        self.poke_and_image = "poke_and_image" in self.config["architecture"] and self.config["architecture"]["poke_and_image"]


        self.vgg_loss = PerceptualLoss()

        # ae
        self.model = FirstStageWrapper(self.config)


        # metrics
        # self.ssim = SSIM()
        # self.psnr = PSNR()
        self.lpips_net = lpips_net()
        for param in self.lpips_net.parameters():
            param.requires_grad = False

        self.lpips_metric = LPIPS()

    def setup(self, stage: str):
        assert isinstance(self.logger, WandbLogger)
        self.logger.experiment.watch(self, log="all")

    def training_step(self, batch, batch_idx):
        if isinstance(batch['poke'], list):
            poke = batch["poke"][0]
            poke_coords = batch["poke"][1]
        else:
            poke = batch['poke']
            poke_coords = None
        flow = batch["flow"]

        if self.poke_and_image:
            img = batch["images"][:, 0]
            poke = torch.cat([poke, img], dim=1)

        poke_in = flow if self.flow_ae else poke

        rec = self.model(poke_in)
        rec_loss = torch.abs(flow.contiguous() - rec.contiguous())

        zeros = torch.zeros((flow.size(0), 1, *flow.shape[-2:]), device=self.device)
        p_loss = self.vgg_loss(torch.cat([flow, zeros], 1).contiguous(), torch.cat([rec, zeros], 1).contiguous())
        # equal weighting of l1 and perceptual loss
        rec_loss = rec_loss + self.perc_weight * p_loss



        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        loss = nll_loss

        loss_dict = {"train/loss": loss, "train/logvar": self.logvar.detach(), "train/nll_loss": nll_loss, }
        self.log_dict(loss_dict, logger=True, on_epoch=True, on_step=True)
        self.log("global step", self.global_step)
        self.log("learning rate", self.optimizers().param_groups[0]["lr"], on_step=True, logger=True)

        self.log("overall_loss", loss, prog_bar=True, logger=False)
        self.log("nll_loss", nll_loss, prog_bar=True, logger=False)

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            flow_orig = batch["original_flow"].detach()
            img = batch["images"][:, 0].detach()
            poke = poke[:, :2].detach()
            flows = [poke.detach(), rec.detach(), flow.detach(), flow_orig.detach()]
            captions = ["Poke", "Flow-rec", "Flow-target", "Flow-orig"]
            if self.flow_ae:
                poke_coords = None
            train_grid_cmap = batches2flow_grid(flows, captions, n_logged=self.n_logged_imgs, img=img, poke=flows[0],
                                                poke_coords=poke_coords,poke_normalized=False)
            train_grid_quiver = batches2flow_grid(flows, captions, n_logged=self.n_logged_imgs, quiver=True, img=img, poke=flows[0],
                                                  poke_coords=poke_coords,poke_normalized=False)
            self.logger.experiment.log({f"Train Batch Cmap": wandb.Image(train_grid_cmap,
                                                                         caption=f"Training Images @ it #{self.global_step}"),
                                        f"Train Batch Quiver": wandb.Image(train_grid_quiver,
                                                                           caption=f"Training Images @ it #{self.global_step}"),
                                        }, step=self.global_step)

        return loss

    def training_epoch_end(self, outputs):
        self.log("epoch", self.current_epoch)

    def validation_step(self, batch, batch_id):
        #poke = batch["poke"][0] if isinstance(batch["poke"], list) else batch["poke"]
        if isinstance(batch['poke'],list):
            poke = batch["poke"][0]
            poke_coords = batch["poke"][1]
        else:
            poke = batch['poke']
            poke_coords=None

        flow = batch["flow"]

        if self.poke_and_image:
            img = batch["images"][:, 0]
            poke = torch.cat([poke, img], dim=1)

        poke_in = flow if self.flow_ae else poke
        with torch.no_grad():
            rec = self.model(poke_in)
            rec_loss = torch.abs(flow.contiguous() - rec.contiguous())

            zeros = torch.zeros((flow.size(0), 1, *flow.shape[-2:]), device=self.device)
            f3 = torch.cat([flow, zeros], 1).contiguous()
            r3 = torch.cat([rec, zeros], 1).contiguous()
            p_loss = self.vgg_loss(f3, r3)
            # equal weighting of l1 and perceptual loss
            rec_loss = rec_loss + self.perc_weight * p_loss


            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
            loss = nll_loss

        log_dict = {"val/loss": loss, "val/logvar": self.logvar.detach(),
                    "val/nll_loss": nll_loss, "val/rec_loss": rec_loss}

        self.log_dict(log_dict, logger=True, prog_bar=False, on_epoch=True)

        # self.log("ssim-val", self.ssim(rec, flow).cpu(), on_step=False, on_epoch=True, logger=True)
        # self.log("psnr-val", self.psnr(rec, flow).cpu(), on_step=False, on_epoch=True, logger=True)
        self.log("lpips-val", self.lpips_metric(self.lpips_net, r3, f3).cpu(), on_step=False, on_epoch=True, logger=True)

        if batch_id < self.config["logging"]["n_val_img_batches"]:
            flow_orig = batch["original_flow"].detach()
            img = batch["images"][:, 0].detach()
            poke = poke[:, :2].detach()
            flows = [poke, rec, flow, flow_orig]
            captions = ["Poke", "Flow-rec", "Flow-target", "Flow-orig"]
            if self.flow_ae:
                poke_coords = None
            val_grid_cmap = batches2flow_grid(flows, captions, n_logged=self.n_logged_imgs, img=img, poke=flows[0],
                                              poke_coords=poke_coords,poke_normalized=False)
            val_grid_quiver = batches2flow_grid(flows, captions, n_logged=self.n_logged_imgs, quiver=True, img=img, poke=flows[0],
                                                poke_coords=poke_coords,poke_normalized=False)
            self.logger.experiment.log({f"Validation Batch #{batch_id} Cmap Plot": wandb.Image(val_grid_cmap,
                                                                                               caption=f"Validation Images @ it {self.global_step}"),
                                        f"Validation Batch #{batch_id} Quiver Plot": wandb.Image(val_grid_quiver,
                                                                                                 caption=f"Validation Images @ it {self.global_step}")
                                        }, step=self.global_step
                                       )

        return log_dict, batch_id

    def configure_optimizers(self):
        # optimizers
        opt_g = Adam(self.parameters(), lr=self.config["training"]["lr"], weight_decay=self.config["training"]["weight_decay"])
        # schedulers
        sched_g = lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=.5, patience=1, min_lr=1e-8,
                                                 threshold=0.0001, threshold_mode='abs')
        return [opt_g], [{'scheduler': sched_g, 'monitor': "loss-val", "interval": 1, 'reduce_on_plateau': True, 'strict': True}, ]
        # return ({'optimizer': opt_g,'lr_scheduler':sched_g,'monitor':"loss-val","interval":1,'reduce_on_plateau':True,'strict':True},
        # {'optimizer': opt_d,'lr_scheduler':sched_d,'monitor':"loss-val","interval":1,'reduce_on_plateau':True,'strict':True})
