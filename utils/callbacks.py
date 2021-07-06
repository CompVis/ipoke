import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from matplotlib import pyplot as plt
import wandb
import umap
from os import path

class BestCkptsToYaml(Callback):
    def __init__(self,ckpt_callback:ModelCheckpoint):
        #super.__init__()
        assert isinstance(ckpt_callback,ModelCheckpoint)
        self.ckpt_cb = ckpt_callback


    def on_train_end(self,trainer, pl_module):
        if path.isdir(self.ckpt_cb.dirpath):
            self.ckpt_cb.to_yaml()

    def on_validation_epoch_end(self, trainer, pl_module):
        if path.isdir(self.ckpt_cb.dirpath):
            self.ckpt_cb.to_yaml()



class UMAP(Callback):
    def __init__(self, batch_frequency, n_samples):
        super().__init__()
        self.batch_freq = batch_frequency
        self.n_samples = n_samples

    def log_umap(self, pl_module, first_stage_data, split="train"):

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        n_samples = self.n_samples // first_stage_data.size(0)
        dloader = pl_module.train_dataloader() if split == 'train' else pl_module.val_dataloader()
        z, z_m, z_p = [], [], []
        while len(z) < n_samples:
            for batch_idx, batch in enumerate(dloader):
                if len(z) > n_samples:
                    break
                with torch.no_grad():
                    seq = batch['seq'].to(pl_module.device)
                    ## Create embeddings from first stage model
                    posterior = pl_module.first_stage_model.encode(seq)
                    z_m.append(posterior.mode().squeeze(-1).squeeze(-1).detach().cpu().numpy())
                    z_p.append(posterior.sample().squeeze(-1).squeeze(-1).detach().cpu().numpy())
                    ## Create embeddings from flow by reversing direction
                    gaussian = torch.randn_like(z_m)
                    embed = pl_module.flow(gaussian, seq[:, 0], reverse=True).squeeze(-1).squeeze(-1)
                    z.append(embed.detach().cpu().numpy())
        z = np.concatenate(z)
        z_m = np.concatenate(z_m)
        z_p = np.concatenate(z_p)
        umap_transform = umap.UMAP()
        transformation = umap_transform.fit(z_m)
        transformed_z = transformation.transform(z_m)
        plt.scatter(transformed_z[:, 0], transformed_z[:, 1], c='blue', s=1, marker='o', label="mean", alpha=.3, rasterized=True)
        plt.scatter(np.mean(transformed_z[:, 0]), np.mean(transformed_z[:, 1]), c='blue', s=20, marker='o', label="mean mean", alpha=.3)
        transformed_z = transformation.transform(z)
        plt.scatter(transformed_z[:, 0], transformed_z[:, 1], c='red', s=1, marker='v', label="INN samples", alpha=.3, rasterized=True)
        plt.scatter(np.mean(transformed_z[:, 0]), np.mean(transformed_z[:, 1]), c='red', s=20, marker='o', label="INN samples mean", alpha=.3)
        transformed_z = transformation.transform(z_p)
        plt.scatter(transformed_z[:, 0], transformed_z[:, 1], c='green', s=1, marker='s', label="posterior", alpha=.3, rasterized=True)
        plt.scatter(np.mean(transformed_z[:, 0]), np.mean(transformed_z[:, 1]), c='green', s=20, marker='o', label="posterior mean", alpha=.3)
        plt.legend()
        plt.axis('off')
        plt.ioff()
        pl_module.logger.experiment.log({"Umap plot " + split: wandb.Image(plt, caption="Umap plot")})
        plt.close()
        if is_train:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_umap(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_umap(pl_module, batch, batch_idx, split="val")