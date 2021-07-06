import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
import wandb
from os import path,makedirs
import yaml
import numpy as np
from functools import partial
from lpips import LPIPS as lpips_net
import os
import cv2
from tqdm import tqdm
import pandas as pd
import time
import math

from models.first_stage_motion_model import SpadeCondMotionModel,RNNMotionModel
from models.pretrained_models import conditioner_models,first_stage_models,poke_embedder_models
from models.modules.autoencoders.fully_conv_models import FirstStageWrapper
from models.modules.INN.INN import SupervisedMacowTransformer, MacowTransformerMultiStep
from models.modules.INN.loss import FlowLoss
from models.modules.autoencoders.util import Conv2dTransposeBlock
from models.modules.INN.coupling_flow_alternative import AdaBelief
from utils.logging import make_flow_video_with_samples, log_umap, make_samples_and_samplegrid, save_video, make_transfer_grids_new, make_multipoke_grid
from utils.general import linear_var, get_logger
from utils.metrics import FVD, calculate_FVD, LPIPS,PSNR_custom,SSIM_custom, KPSMetric, metric_vgg16, compute_div_score,SampleLPIPS, SampleSSIM, compute_div_score_mse, compute_div_score_lpips
from utils.posenet_wrapper import PoseNetWrapper
from models.pose_estimator.tools.infer import save_batch_image_with_joints

class PokeMotionModel(pl.LightningModule):

    def __init__(self,config,dirs):
        super().__init__()
        #self.automatic_optimization=False
        self.config = config
        self.embed_poke = True
        self.dirs = dirs
        self.test_mode = self.config['general']['test']
        self.spatial_mean_for_loss = 'spatial_mean' in self.config['training'] and self.config['training']['spatial_mean']
        logdet_weight = self.config['training']['logdet_weight'] if 'logdet_weight' in self.config else 1.
        self.use_adabelief = 'adabelief' in self.config['training'] and self.config['training']['adabelief']
        self.n_test_samples = self.config['testing']['n_samples_per_data_point']


        self.console_logger = get_logger()

        # configure learning rate scheduling, if intended
        self.apply_lr_scaling = "lr_scaling" in self.config["training"] and self.config["training"]["lr_scaling"]
        if self.apply_lr_scaling:
            end_it = self.config["training"]["lr_scaling_max_it"]
            lr = self.config["training"]["lr"]
            self.lr_scaling = partial(linear_var, start_it=0, end_it=end_it, start_val=0., end_val=lr, clip_min=0.,
                                      clip_max=lr)

        #self.embed_poke_and_image = self.config["poke_embedder"]["embed_poke_and_image"]
        self.use_flow_as_poke = False
        self.__initialize_first_stage()

        self.full_seq = 'full_seq' in self.config['training'] and self.config['training']['full_seq']

        self.use_cond = self.config['conditioner']['use'] if 'use' in self.config['conditioner'] else True
        if self.use_cond:
            self.__initialize_conditioner()

        self.augment_input = 'augmented_input' in self.config['architecture'] and self.config['architecture']['augmented_input']
        self.config["architecture"]["flow_in_channels"] = self.first_stage_config["architecture"]["z_dim"]
        if self.augment_input:
            self.config['architecture']['flow_in_channels'] += self.config['architecture']['augment_channels']
            if 'scale_augmentation' in self.config['architecture'] and self.config['architecture']['scale_augmentation']:
                self.scale_augment = nn.Parameter(torch.ones(self.config['architecture']['augment_channels']),requires_grad=True)
            else:
                self.register_buffer('scale_augment',torch.ones(self.config['architecture']['augment_channels']))


            if 'shift_augmentation' in self.config['architecture'] and self.config['architecture']['shift_augmentation']:
                self.shift_augment = nn.Parameter(torch.zeros(self.config['architecture']['augment_channels']),requires_grad=True)
            else:
                self.register_buffer('shift_augment',torch.zeros(self.config['architecture']['augment_channels']))
        # get conditioning size

        self.metrics_dir = path.join(self.dirs['generated'],'metrics')
        os.makedirs(self.metrics_dir,exist_ok=True)


        self.FVD = FVD(n_samples=self.config['logging']['n_fvd_samples'] if 'n_fvd_samples' in  self.config['logging'] else 1000)
        if self.test_mode == 'none' or self.test_mode=='accuracy':
            self.lpips_metric = LPIPS()
            self.ssim = SSIM_custom()
            self.psnr = PSNR_custom()
            self.lpips_net = lpips_net()



        self.__initialize_poke_embedder()
        self.embed_poke_and_image = "poke_and_image" in self.poke_emb_config["architecture"] and self.poke_emb_config["architecture"]["poke_and_image"]
        self.poke_key = 'flow' if self.poke_emb_config['architecture']['flow_ae'] else "poke"
        if self.poke_key =='flow':
            assert not self.embed_poke_and_image
        if self.use_cond:
            self.config["architecture"]["h_channels"] = self.conditioner_config["architecture"]["nf_max"] + \
                                                                self.poke_emb_config["architecture"]["nf_max"]
        else:
            self.config["architecture"]["h_channels"] =self.poke_emb_config["architecture"]["nf_max"]


        self.config["architecture"]["flow_mid_channels"] = int(config["architecture"]["flow_mid_channels_factor"] * \
                                                               self.config["architecture"]["flow_in_channels"])

        self.config['architecture'].update({'ssize': self.poke_emb_config['architecture']['min_spatial_size']})
        model = MacowTransformerMultiStep if 'multistack' in self.config['architecture'] and self.config['architecture']['multistack'] else SupervisedMacowTransformer


        self.adapt_poke_emb_ssize = self.poke_emb_config['architecture']['min_spatial_size'] != self.first_stage_config['architecture']['min_spatial_size']
        if self.adapt_poke_emb_ssize:
            factor = float(self.first_stage_config['architecture']['min_spatial_size']) / self.poke_emb_config['architecture']['min_spatial_size']
            self.conv_adapt_poke_emb = nn.Conv2d(self.poke_emb_config['architecture']['nf_max'],self.poke_emb_config['architecture']['nf_max'],stride=int(factor),kernel_size=3,padding=1) if factor > 1 else \
                Conv2dTransposeBlock(self.poke_emb_config['architecture']['nf_max'],self.poke_emb_config['architecture']['nf_max'],ks=3,st=int(1./factor),padding=1,norm='group') #,

        self.adapt_cond_ssize = self.use_cond and self.conditioner_config['architecture']['min_spatial_size'] != \
                                    self.first_stage_config['architecture']['min_spatial_size']
        if self.adapt_cond_ssize:
            factor = float(self.first_stage_config['architecture']['min_spatial_size']) / \
                     self.conditioner_config['architecture']['min_spatial_size']
            self.conv_adapt_cond = nn.Conv2d(self.conditioner_config['architecture']['nf_max'],
                                                 self.conditioner_config['architecture']['nf_max'],
                                                 stride=int(factor),kernel_size=3,padding=1) if factor < 1 else \
                Conv2dTransposeBlock(self.conditioner_config['architecture']['nf_max'],
                                   self.conditioner_config['architecture']['nf_max'], st=int(factor),ks=3,padding=1)

        self.flow = model(self.config["architecture"])


        n_samples_umap = self.config["logging"]["n_samples_umap"] if "n_samples_umap" in self.config["logging"] else 1000
        self.n_it_umap = int(math.ceil(n_samples_umap / self.config["data"]["batch_size"])) if not self.config['general']['debug'] else 2
        samples_dict= {"z_s":[],"z_p":[],"z_m":[]}

        self.log_samples = {"train": samples_dict.copy(), "val": samples_dict.copy()}

        self.loss_func = FlowLoss(spatial_mean=self.spatial_mean_for_loss,logdet_weight=logdet_weight)
        self.apply_lr_scaling = "lr_scaling" in self.config["training"] and self.config["training"]["lr_scaling"]
        lr = self.config["training"]["lr"]
        if self.apply_lr_scaling:
            end_it = self.config["training"]["lr_scaling_max_it"]

            self.lr_scaling = partial(linear_var, start_it=0, end_it=end_it, start_val=0., end_val=lr, clip_min=0., clip_max=lr)

        self.custom_lr_decrease = self.config['training']['custom_lr_decrease']
        if self.custom_lr_decrease:
            start_it = self.config["training"]["lr_scaling_max_it"]  # 1000
            self.lr_adaptation = partial(linear_var, start_it=start_it, start_val=lr, end_val=0., clip_min=0.,
                                         clip_max=lr)

        if self.test_mode == 'diversity':
            self.vggm = metric_vgg16()
            self.div_scores = []
            self.posenet = PoseNetWrapper(self.config)


        if self.test_mode == 'accuracy':
            # instantiate required metrics
            self.sample_lpips = SampleLPIPS(self.console_logger,self.config['testing']['n_samples_per_data_point'])
            # self.sample_psnr = SamplePSNR(self.console_logger,self.config['testing']['n_samples_per_data_point'])
            self.sample_ssim = SampleSSIM(self.console_logger, self.config['testing']['n_samples_per_data_point'])

            self.posenet = PoseNetWrapper(self.config)
            # self.savedir_pose_plots = path.join(self.dirs['generated'],'kps_experiments','nn_var_experiment')
            self.pose_nn_metric = KPSMetric(self.console_logger,n_samples=self.config['testing']['n_samples_metrics'])
            self.metrics_dict = {'KPS':{},'LPIPS':{},'SSIM':{},'PSNR':{},}

        if self.test_mode =='kps_acc':
            self.posenet = PoseNetWrapper(self.config)
        self.start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.transfer_dir = path.join(self.dirs['generated'],'transfer',self.start_time)
        os.makedirs(self.transfer_dir,exist_ok=True)

        if self.test_mode == 'fvd':
            self.savedir_fvd = path.join(self.dirs['generated'], 'samples_fvd')



    def __initialize_first_stage(self):
        dic = first_stage_models[self.config['first_stage']['name']]
        model_name = dic['model_name']
        first_stage_ckpt = dic['ckpt']


        first_stage_config = path.join(self.config["general"]["base_dir"], "first_stage", "config", model_name, "config.yaml")

        with open(first_stage_config) as f:
            self.first_stage_config = yaml.load(f, Loader=yaml.FullLoader)


        #strict = False, because no need to load parameters of discriminators etc
        self.first_stage_model = SpadeCondMotionModel.load_from_checkpoint(first_stage_ckpt,config=self.first_stage_config,
                                                                        train=False,strict=False,dirs=self.dirs)

        if self.first_stage_model.enc_motion.be_determinstic:
            self.console_logger.info('First stage model is deterministic...')

    def __initialize_conditioner(self):
        dic = conditioner_models[self.config['conditioner']['name']]
        model_name = dic['model_name']
        conditioner_ckpt = dic['ckpt']
        conditioner_config = path.join(self.config["general"]["base_dir"], "img_encoder", "config",
                                       model_name, "config.yaml")


        with open(conditioner_config) as f:
            self.conditioner_config = yaml.load(f, Loader=yaml.FullLoader)

        self.conditioner = FirstStageWrapper(self.conditioner_config)
        if 'restart' in self.config['general'] and not self.config["general"]["restart"]:
            state_dict = torch.load(conditioner_ckpt, map_location="cpu")
            # remove keys from checkpoint which are not required
            state_dict = {key: state_dict["state_dict"][key] for key in state_dict["state_dict"] if
                          "encoder" in key or "decoder" in key}
            # load first stage model
            self.conditioner.load_state_dict(state_dict, strict=False)

    def __initialize_poke_embedder(self):
        dic = poke_embedder_models[self.config['poke_embedder']['name']]
        model_name = dic['model_name']
        emb_ckpt = dic['ckpt']

        emb_config = path.join(self.config["general"]["base_dir"], "poke_encoder", "config", model_name, "config.yaml")

        with open(emb_config) as f:
            self.poke_emb_config = yaml.load(f, Loader=yaml.FullLoader)
        self.poke_embedder = FirstStageWrapper(self.poke_emb_config)
        assert self.poke_embedder.be_deterministic
        state_dict = torch.load(emb_ckpt, map_location="cpu")
        # remove keys from checkpoint which are not required
        state_dict = {".".join(key.split(".")[1:]): state_dict["state_dict"][key] for key in state_dict["state_dict"] if "encoder" in key or "decoder" in key}
        # load first stage model
        self.poke_embedder.load_state_dict(state_dict)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.apply_lr_scaling and self.global_step < self.config["training"]["lr_scaling_max_it"]:
            # adjust learning rate
            lr = self.lr_scaling(self.global_step)
            opt = self.optimizers()
            self.log("learning_rate",lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)
            for pg in opt.param_groups:
                pg["lr"] = lr


        if self.custom_lr_decrease and self.global_step >= self.config["training"]["lr_scaling_max_it"]:
            lr = self.lr_adaptation(self.global_step)
            # self.console_logger.info(f'global step is {self.global_step}, learning rate is {lr}\n')
            opt = self.optimizers()
            for pg in opt.param_groups:
                pg["lr"] = lr

    def make_flow_input(self,batch,reverse=False,use_kp_poke=False):
        X = batch['images']

        if use_kp_poke:
            poke, *_ = batch['keypoint_poke']
            poke = poke.to(torch.float)
        else:
            poke = batch[self.poke_key]
            poke = poke[0] if isinstance(poke, list) else poke

        if self.embed_poke_and_image:
            poke = torch.cat([poke,X[:,0]],dim=1)

        # always eval
        self.first_stage_model.eval()
        self.poke_embedder.eval()
        if self.use_cond:
            self.conditioner.eval()
        with torch.no_grad():
            poke_emb, *_ = self.poke_embedder.encoder(poke)
            # do not sample, as this mapping should be deterministic
            if self.adapt_poke_emb_ssize:
                poke_emb = self.conv_adapt_poke_emb(poke_emb)

            if self.use_cond:
                if self.conditioner.be_deterministic:
                    cond, *_ = self.conditioner.encoder(X[:,0])
                else:
                    _, cond, _ = self.conditioner.encoder(X[:,0])


                if self.adapt_cond_ssize:
                    cond = self.conv_adapt_cond(cond)

        if reverse:
            if self.flow.flow.reshape != "none":
                cn_factor = 4 if self.flow.flow.reshape == 'down' else 1/4
                spatial_factor = 2 if self.flow.flow.reshape == 'down' else .5
                shape = [X.size(0), int(cn_factor * self.config['architecture']['flow_in_channels']),
                         int(self.first_stage_config['architecture']['min_spatial_size'] / spatial_factor),
                         int(self.first_stage_config['architecture']['min_spatial_size'] / spatial_factor)]
                flow_input = torch.randn(shape).type_as(X)
            else:
                #flow_input, *_ = self.encode_first_stage(X)
                spatial=self.first_stage_config['architecture']['min_spatial_size']
                flow_input = torch.randn((X.size(0),self.config['architecture']['flow_in_channels'],spatial,spatial)).type_as(X).detach()
        else:
            with torch.no_grad():
                flow_input, *_ = self.encode_first_stage(X)
            if self.augment_input:
                # augment input with random noise, similar to idea of augmented normalizing odes
                input_augment = torch.randn((flow_input.size(0),self.config['architecture']['augment_channels'],*flow_input.shape[-2:])).type_as(X)
                input_augment = self.scale_augment[None,:,None,None] * input_augment + self.shift_augment[None,:,None,None]
                flow_input = torch.cat([flow_input,input_augment],dim=1)

        if self.use_cond:
            cond = torch.cat([cond, poke_emb], dim=1)
        else:
            cond = poke_emb

        return flow_input, cond

    def on_train_epoch_start(self):
        # n_overall_batches = self.trainer.num_val_batches[-1]
        # self.n_it_fid = min(self.n_it_fid,n_overall_batches)

        if self.custom_lr_decrease:
            n_train_iterations = self.config['training']['n_epochs'] * self.trainer.num_training_batches
            self.lr_adaptation = partial(self.lr_adaptation, end_it=n_train_iterations)


    def forward_sample(self, batch, n_samples=1, n_logged_vids=1, show_progress= False, add_first_frame=False,use_keypoint_pokes=False):
        video_samples = []

        with torch.no_grad():
            X = batch['images']
            it = tqdm(range(n_samples),desc=f'Generating {n_samples} samples') if show_progress else range(n_samples)
            for n in it:
                flow_input, cond = self.make_flow_input(batch, reverse=True,use_kp_poke=use_keypoint_pokes)
                out_motion = self.flow(flow_input, cond, reverse=True)
                if self.augment_input:
                    out_motion = out_motion[:,:-self.config['architecture']['augment_channels']]
                out_video = self.decode_first_stage(out_motion,X)
                if add_first_frame:
                    out_video = torch.cat([X[:,0].unsqueeze(1),out_video],dim=1)

                video_samples.append(out_video[:n_logged_vids].cpu())

        return video_samples

    def forward_density(self, batch):
        flow_input, cond = self.make_flow_input(batch)

        out, logdet = self.flow(flow_input.detach(), cond, reverse=False)

        return out, logdet

    def encode_first_stage(self,X):
        with torch.no_grad():
            if self.full_seq:
                X_in = X if self.first_stage_model.full_sequence or self.config['data']['max_frames'] < 16 else X[:, :-1]
            else:
                X_in = X if self.first_stage_model.full_sequence else X[:, 1:]
            motion, mu, cov = self.first_stage_model.enc_motion(X_in.transpose(1, 2))
        return motion, mu

    def decode_first_stage(self,motion,X, length=None):

        if isinstance(self.first_stage_model,SpadeCondMotionModel):
            start_frame = X[:,0]
            hidden = [motion] * self.first_stage_model.n_layers
            # x = X[:,0]

            X_hat = []
            if self.first_stage_model.use_motion_bias:
                in_rnn = torch.cat([self.first_stage_model.motion_bias] * start_frame.size(0), dim=0)
            else:
                in_rnn = motion
            if length is None:
                length = X.size(1) - 1
            for i in range(length):
                hidden = self.first_stage_model.rnn(in_rnn, hidden)

                reaction = self.first_stage_model.gen([hidden[-1]], start_frame, del_shape=True)

                X_hat.append(reaction)

            X_hat = torch.stack(X_hat, dim=1)

        elif isinstance(self.first_stage_model, RNNMotionModel):
            scene = self.first_stage_model.enc_static(X[:, 0])[0]
            motion, mu, cov = self.first_stage_model.enc_motion(X[:, 1:].transpose(1, 2))

            # hidden state is initiazed with motion encoding
            hidden = [motion] * self.n_layers
            x = scene

            X_hat = []
            for i in range(X.size(1) - 1):
                hidden = self.first_stage_model.rnn(x, hidden)
                x = self.first_stage_model.post_hidden(hidden[-1])
                reaction = self.first_stage_model.gen([x], del_shape=True)

                X_hat.append(reaction)

            X_hat = torch.stack(X_hat, dim=1)

            return X_hat, mu, cov
        else:
            X_hat = self.gen(X[:, 0], motion)

        return X_hat


    def training_step(self,batch, batch_idx):



        out, logdet = self.forward_density(batch)

        loss, loss_dict = self.loss_func(out,logdet)

        self.log_dict(loss_dict,prog_bar=True,on_step=True,logger=False)
        self.log_dict({"train/"+key: loss_dict[key] for key in loss_dict},logger=True,on_epoch=True,on_step=True)
        self.log("global_step",self.global_step, on_epoch=False,logger=True,on_step=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            n_samples = self.config["logging"]["n_samples"]
            n_logged_vids = self.config["logging"]["n_log_images"]

            vid_samples = self.forward_sample(batch,n_samples,n_logged_vids)

            x_0 = batch["images"][:n_logged_vids ,0]
            X_tgt = batch["images"][:n_logged_vids]

            poke = batch["poke"]
            if isinstance(poke,list):
                poke_coords = poke[1][:n_logged_vids]
                poke = poke[0][:n_logged_vids]
            else:
                poke_coords = None
            # poke = poke[0] if isinstance(poke,list) else poke
            flow = batch["flow"][:n_logged_vids]

            animated_grid = make_flow_video_with_samples(x_0.detach().cpu(),poke.detach().cpu(),vid_samples,
                                                         X_tgt.detach().cpu(),flow.detach().cpu(),
                                                         n_logged=n_logged_vids,
                                                         poke_coords=poke_coords)
            self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({"Flow Video Grid train set": wandb.Video(animated_grid,
                                                                                    caption=f"Flow Video Grid train @ it #{self.global_step}",
                                                                                    fps=3,format="mp4")},
                                       step=self.global_step, commit=False)

        return loss

    def __make_umap_samples(self,key,batch):
        X = batch["images"]
        with torch.no_grad():
            # posterior
            z_p, z_m = self.encode_first_stage(X)

            if self.augment_input:
                # scale samples with samples
                input_augment = torch.randn(
                    (z_p.size(0), self.config['architecture']['augment_channels'], *z_p.shape[-2:]),
                    device=self.device)
                input_augment_p = self.scale_augment[None, :, None, None] * input_augment + self.shift_augment[None,:,None,None]
                z_p = torch.cat([z_p,input_augment_p],dim=1)
                # add the mean to the means, which is zero for the augmented space
                z_m = torch.cat([z_m,self.shift_augment],dim=1)


            self.log_samples[key]["z_m"].append(z_m.detach().cpu().numpy())
            self.log_samples[key]["z_p"].append(z_p.detach().cpu().numpy())
            # from residual
            flow_input, cond = self.make_flow_input(batch, reverse=True)
            z_s = self.flow(flow_input, cond, reverse=True)
            if not torch.isnan(z_s).any():
                self.log_samples[key]["z_s"].append(z_s.detach().cpu().numpy())
                return 0
            else:
                self.console_logger.info("NaN encountered in umap samples.")
                return 1

    def training_epoch_end(self, outputs):
        self.log("epoch",self.current_epoch)

        if self.current_epoch % 3 == 0:
            self.log_umap(train=True)


    def validation_step(self, batch, batch_id):

        out, logdet = self.forward_density(batch)

        loss, loss_dict = self.loss_func(out, logdet)

        self.log_dict({"val/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True)

        X = batch['images']
        if batch_id <= int(self.config["logging"]["n_fvd_samples"] / X.size(0)):
            X_hat = self.forward_sample(batch,n_logged_vids=X.size(0))[0]
            #self.FVD.update(X_hat,X)
            self.first_stage_model.features_fvd_fake.append(X_hat.cpu().numpy())
            self.first_stage_model.features_fvd_true.append(X[:, 1:].cpu().numpy())

            self.first_stage_model.fvd_features_fake_x0.append(np.concatenate([X[:, 0].unsqueeze(1).cpu().numpy(), X_hat.numpy()], axis=1))
            self.first_stage_model.fvd_features_true_x0.append(X.cpu().numpy())


            X_hat_log = X_hat.reshape(-1, *X_hat.shape[2:]).type_as(X)
            X_log = X[:, 1:].reshape(-1, *X_hat.shape[2:])
            self.log("ssim-val", self.ssim(X_hat_log, X_log).cpu(), on_step=False, on_epoch=True, logger=True)
            self.log("psnr-val", self.psnr(X_hat_log, X_log).cpu(), on_step=False, on_epoch=True, logger=True)
            self.log("lpips-val", self.lpips_metric(self.lpips_net, X_hat_log, X_log).cpu(), on_step=False, on_epoch=True, logger=True)


        return {"loss":loss, "val-batch": batch, "batch_idx": batch_id, "loss_dict":loss_dict}

    def validation_step_end(self,outputs):
        batch_idx = outputs["batch_idx"]
        loss = outputs["loss"]
        loss_dict = outputs["loss_dict"]

        self.log("d_ref_nll-val", torch.abs(loss_dict["reference_nll_loss"] - loss_dict["nll_loss"]), on_epoch=True, logger=True)
        self.log("loss-val",loss,on_epoch=True,logger=True)


        if batch_idx < self.config["logging"]["n_val_img_batches"]:
            self.eval()
            batch = outputs["val-batch"]

            n_samples = self.config["logging"]["n_samples"]
            n_logged_vids = self.config["logging"]["n_log_images"]

            vid_samples = self.forward_sample(batch,n_samples,n_logged_vids)

            x_0 = batch["images"][:n_logged_vids,0]
            X_tgt = batch["images"][:n_logged_vids]
            poke = batch["poke"]
            if isinstance(poke, list):
                poke_coords = poke[1][:n_logged_vids]
                poke = poke[0][:n_logged_vids]
            else:
                poke_coords = None
            flow = batch["flow"][:n_logged_vids]



            animated_grid = make_flow_video_with_samples(x_0.detach().cpu(),poke.detach().cpu(),vid_samples,
                                                         X_tgt.detach().cpu(),flow.detach().cpu(),
                                                         n_logged=n_logged_vids,
                                                         poke_coords=poke_coords)
            self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({f"Animated Flow Grid val set": wandb.Video(animated_grid,
                                                                                                caption=f"Animated grid val #{batch_idx}",
                                                                                                fps=3,format="mp4")},
                                       step=self.global_step, commit=False)

    def validation_epoch_end(self, outputs):
        #fvd_score = self.FVD.compute()
        self.FVD.i3d.eval()
        features_fake = torch.from_numpy(np.concatenate(self.first_stage_model.features_fvd_fake, axis=0))
        features_true = torch.from_numpy(np.concatenate(self.first_stage_model.features_fvd_true, axis=0))

        fvd_score = calculate_FVD(self.FVD.i3d, features_fake, features_true,
                                   batch_size=self.first_stage_config["logging"]["bs_i3d"], cuda=True)

        self.log('FVD-val',fvd_score,on_epoch=True,on_step=False,logger=True)
        self.console_logger.info(f'FVD score after epoch {self.current_epoch} is {fvd_score}')

        features_fake = torch.from_numpy(np.concatenate(self.first_stage_model.fvd_features_fake_x0, axis=0))
        features_true = torch.from_numpy(np.concatenate(self.first_stage_model.fvd_features_true_x0, axis=0))

        fvd_score = calculate_FVD(self.FVD.i3d, features_fake, features_true,
                                  batch_size=self.first_stage_config["logging"]["bs_i3d"], cuda=True)

        self.log('FVD-val-x0', fvd_score, on_epoch=True, logger=True)

        self.first_stage_model.features_fvd_fake.clear()
        self.first_stage_model.features_fvd_true.clear()
        self.first_stage_model.fvd_features_fake_x0.clear()
        self.first_stage_model.fvd_features_true_x0.clear()

        if self.current_epoch % 3 == 0:
            self.log_umap()



    def log_umap(self, train = False):

        if train:
            dloader = self.train_dataloader()
            key = "train"
        else:
            dloader = self.val_dataloader()
            key = "val"

        n_nans = 0

        for batch_idx, batch in enumerate(tqdm(dloader,desc=f"Logging Umap Plot on {key} data")):
            if batch_idx >= self.n_it_umap:
                break
            batch = {k: batch[k][0].to(self.device) if isinstance(batch[k],list) else batch[k].to(self.device) for k in batch}
            n_nans +=self.__make_umap_samples(key,batch)

        self.console_logger.info(f'In total {n_nans} batches, containing NaNs while preparing samples for UMap plot...')


        if len(self.log_samples[key]["z_s"]) >1:

            z_s = np.concatenate(self.log_samples[key]["z_s"])
            n_data_points = z_s.shape[0]
            z_s = z_s.reshape(n_data_points, -1)
        else:
            # reset log_samples dict
            self.log_samples[key] = {k: [] for k in self.log_samples[key]}
            self.console_logger.warn(f'NO SINGLE VALID BATCH FOR LOGGING UMAP in "{key}"-MODE. ABORT LOGGING UMAP.')
            return
        len_zm = len(self.log_samples[key]["z_m"])
        z_m = np.concatenate(self.log_samples[key]["z_m"]).reshape(len_zm * self.log_samples[key]["z_m"][0].shape[0], -1)
        len_zp = len(self.log_samples[key]["z_p"])
        z_p = np.concatenate(self.log_samples[key]["z_p"]).reshape(len_zp * self.log_samples[key]["z_p"][0].shape[0], -1)


        log_umap(z_s,z_m,z_p,self.logger,self.global_step,f"Umap plot {key} data")

        # reset log_samples dict
        self.log_samples[key] = {k: [] for k in self.log_samples[key]}




    def configure_optimizers(self):
        trainable_params = [{"params": self.flow.parameters(), "name": "flow"}, ]
        if self.augment_input and isinstance(self.scale_augment,nn.Parameter):
            trainable_params.append({'params': self.scale_augment,'name': 'scale_augment'})

        if self.augment_input and isinstance(self.shift_augment,nn.Parameter):
            trainable_params.append({'params': self.shift_augment, 'name': 'shift_augment'})

        if self.adapt_poke_emb_ssize:
            trainable_params.append({'params': self.conv_adapt_poke_emb.parameters(), 'name': 'adapt_poke_emb'})

        if self.adapt_cond_ssize:
            trainable_params.append({'params': self.conv_adapt_cond.parameters(), 'name': 'adapt_cond'})
        # if self.embed_poke:
        #     trainable_params.append({"params": self.poke_embedder.parameters(), "name": "poke_embedder"})

        optim_type = AdaBelief if self.use_adabelief else Adam
        optimizer = optim_type(trainable_params, lr=self.config["training"]['lr'], betas=(0.9, 0.999),
                         weight_decay=self.config["training"]['weight_decay'], amsgrad=True)
        # optimizer = RMSprop(trainable_params, lr=self.config["training"]['lr'],
        #                  weight_decay=self.config["training"]['weight_decay'],alpha=0.9)
        if "gamma" not in self.config["training"] and not self.custom_lr_decrease:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            return [optimizer], [scheduler]
        elif not self.custom_lr_decrease:
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.config["training"]["gamma"],
                                                   last_epoch=self.current_epoch - 1)
            return [optimizer], [scheduler]

        else:
            return [optimizer]


    def _test_step_fvd(self,batch):
        self.eval()


        with torch.no_grad():
            X = batch['images']
            sample = self.forward_sample(batch,n_logged_vids=batch['images'].size(0))[0]

            out_real = ((X + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
            out_gen = torch.cat([X[:, 0].unsqueeze(1).cpu(), sample], dim=1)
            out_gen = ((out_gen + 1.) * 127.5).permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)

            self.first_stage_model.fvd_features_fake_x0.append(out_gen)
            self.first_stage_model.fvd_features_true_x0.append(out_real)


    def _test_step_metrics(self,batch, batch_id):

        samples = self.forward_sample(batch, self.n_test_samples,
                                      n_logged_vids=self.config['testing']['test_batch_size'],
                                      show_progress=self.config['testing']['verbose'],
                                      add_first_frame=True)

        #samples = torch.stack(samples,dim=1)
        imgs = batch['images']
        if 'keypoints_rel' in batch and 'keypoints_abs' in batch:
            kps_gt = batch['keypoints_rel'][:,None].to(torch.float32)
            kps_abs = batch['keypoints_abs']



            #samples_in = samples.reshape(samples.shape[0],-1,*samples.shape[3:]).type_as(kps_gt)
            kps_pred_rel = []

            for n_sample,sample in enumerate(samples):
                orig_shape = sample.shape
                kps_act_raw = self.posenet(sample.reshape(-1,*sample.shape[2:]).type_as(kps_gt))
                kps_pred_abs, kps_pred_act_rel = self.posenet.postprocess(kps_act_raw)
                kps_pred_rel.append(torch.from_numpy(kps_pred_act_rel.reshape(*orig_shape[:2],*kps_pred_act_rel.shape[1:])))

                #self.console_logger.info(f'batch_id is {batch_id}')
                if self.config['testing']['debug'] and n_sample == 0 and batch_id < 20:

                    n_saved = 2 * orig_shape[1]

                    imgs_with_gt = save_batch_image_with_joints(imgs.reshape(-1,*imgs.shape[2:])[:n_saved], kps_abs.reshape(-1, *kps_abs.shape[2:])[:n_saved], [], None, nrow=n_saved, return_image=True)
                    pred_imgs_with_pred = save_batch_image_with_joints(sample.reshape(-1,*sample.shape[2:])[:n_saved],kps_pred_abs[:n_saved], [], None, nrow=n_saved, return_image=True)

                    grid = np.concatenate([imgs_with_gt, pred_imgs_with_pred], axis=0)
                    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                    savepath = path.join(self.dirs['generated'], 'debug', )
                    os.makedirs(savepath, exist_ok=True)
                    cv2.imwrite(path.join(savepath, f'pose_exmpl_{batch_id}.png'), grid)


            kps_pred_rel = torch.stack(kps_pred_rel,dim=1).type_as(kps_gt)
            self.pose_nn_metric.update(kps_pred_rel, kps_gt)




        target_vid = imgs[:, 1:].unsqueeze(1)
        samples = torch.stack(samples,dim=1).cpu()


        for n_sample in range(samples.shape[1]):
            samples_act = samples[:,n_sample,1:].reshape(-1,*samples.shape[3:]).type_as(target_vid)
            target_act = target_vid.reshape(-1,*target_vid.shape[3:])

            act_ssim = self.ssim(samples_act,target_act).cpu()
            act_psnr = self.psnr(samples_act,target_act).cpu()
            act_lpips = self.lpips_metric(self.lpips_net, samples_act, target_act).cpu()
            if n_sample == 0 and self.config['testing']['verbose']:
                self.console_logger.info(f'batch-ssim: {act_ssim},batch-psnr: {act_psnr},batch-lpips: {act_lpips},')


        self.sample_ssim.update(samples[:,:,1:],target_vid)
        # self.sample_psnr.update(samples[:, :, 1:],target_vid)
        self.sample_lpips.update(samples[:, :, 1:],target_vid)


    def _test_step_diversity(self,batch,batch_id):
        samples = self.forward_sample(batch, self.n_test_samples, n_logged_vids=self.config['testing']['test_batch_size'],
                                      show_progress=self.config['testing']['verbose'])
        samples = torch.stack(samples,dim=1)
        # self.test_aggr.append(samples)

        return samples

    def test_step_end(self, out_step):
        return out_step





    def _test_step_kps_acc(self,batch,batch_id):
        kp_poke, poke_coords, poke_ids = batch['keypoint_poke']
        kp_target = batch['keypoints_rel'][:,-1,].to(torch.float)
        sample = self.forward_sample(batch, 1, n_logged_vids=self.config['testing']['test_batch_size'],
                                      show_progress=False,
                                     use_keypoint_pokes=True)[0]

        kps_act_raw = self.posenet(sample[:,-1].type_as(kp_target))
        kps_pred_abs, kps_pred = self.posenet.postprocess(kps_act_raw)

        # only compute errors wrt the locations of the pokes
        errs = []
        errs_mse = []
        for idx,kpp,kpt in zip(poke_ids,kps_pred,kp_target):
            idx = idx[idx>=0].cpu().numpy()
            kp_dist = kpp[idx]-kpt.cpu().numpy()[idx]

            errs_mse.append(torch.nn.MSELoss()(torch.from_numpy(kpp[idx]),kpt[idx].cpu()))
            errs.append((np.linalg.norm(kp_dist,axis=-1)**2).mean())

        err_mean = np.stack(errs)
        err_mse_mean = torch.tensor(errs_mse)
        return err_mean, err_mse_mean


    def _control_sensitivity(self, batch, batch_id):
        start_id = batch['sample_ids'][:, 0].cpu().numpy()
        flow_map = batch['flow']
        x_0s = batch["images"][:, 0]
        X_tgts = batch["images"]
        pokes = batch["poke"]
        poke_coords = pokes[1]
        pokes = pokes[0]
        half_poke_size = int(self.test_dataloader().dataset.poke_size // 2)

        n_sampled_pokes = self.config['testing']['n_control_sensitivity_pokes']

        amplitude = torch.norm(flow_map, 2, dim=1)
        ampl_min = amplitude.min(-1)[0].min(-1)[0][...,None,None]
        ampl_max = amplitude.max(-1)[0].max(-1)[0][...,None,None]
        amplitude -= ampl_min
        amplitude /= ampl_max


        randomized_pokes = []

        for elem_id,ampl in enumerate(amplitude):
            valid_id = torch.gt(ampl.cpu(),ampl.cpu().mean()).nonzero(as_tuple=True)
            query_ids = torch.randint(valid_id[0].shape[0],size=(n_sampled_pokes,))
            # here, we have shape (n_sampled_pokes,2)
            valid_ids = torch.stack([valid_id[0][query_ids],valid_id[1][query_ids]],dim=1).to(self.device)
            pokes_for_exmpl = []
            # complete_poke_coords_for_exmpl = []
            for id in valid_ids:
                randomized = torch.zeros_like(flow_map[elem_id])

                phase = torch.norm(flow_map[elem_id,:,id[0],id[1]])
                angle = math.pi * torch.rand(size=(1,)).to(self.device)

                sampled_poke = torch.tensor([torch.cos(angle)*phase,
                                             torch.sin(angle)*phase])[...,None,None].type_as(phase)

                # randomized[:,
                # poke_coords[elem_id,0,0]-half_poke_size:poke_coords[elem_id,0,0]+half_poke_size+1,
                # poke_coords[elem_id,0,1]-half_poke_size:poke_coords[elem_id,0,1]+half_poke_size+1] = flow_map[elem_id,:,id[0],id[1]][:,None,None]
                randomized[:,
                poke_coords[elem_id, 0, 0] - half_poke_size:poke_coords[elem_id, 0, 0] + half_poke_size + 1,
                poke_coords[elem_id, 0, 1] - half_poke_size:poke_coords[elem_id, 0, 1] + half_poke_size + 1] = sampled_poke

                pokes_for_exmpl.append(randomized)
                # complete_poke_coords_for_exmpl.append(torch.cat([poke_coords[elem_id,:2],flow_map[elem_id,:,id[0],id[1]]],dim=1))

            randomized_pokes.append(torch.stack(pokes_for_exmpl))
            # complete_poke_coords.append(torch.stack(complete_poke_coords_for_exmpl))

        # shape (n_sampled_pokes,batch_size,2,h,w)
        randomized_pokes = torch.stack(randomized_pokes,dim=1)

        # shape (n_sampled_pokes+1,batch_size,2,h,w)
        randomized_pokes = torch.cat([pokes[None],randomized_pokes])

        samples = []

        for rpoke in tqdm(randomized_pokes, desc=f'Generating samples for randomized pokes, for {batch_id}th batch.'):
            batch['poke'] = rpoke
            rpoke_samples = self.forward_sample(batch,1,n_logged_vids=self.config['testing']['test_batch_size'],
                                                show_progress=False)

            samples.append(rpoke_samples[0])

        # shape (batch_size, n_sampled_pokes+1, t, 3, h, w)
        samples = torch.stack(samples,dim=1)

        # shape (batch_size,n_sampled_pokes+1,n_max_pokes,2)
        # as poke coords are the poke centers and this is the same for all smapled pokes, we can repeat them
        complete_poke_coords = torch.stack([poke_coords]*(n_sampled_pokes+1),dim=1)

        if self.config['general']['last_ckpt']:
            samples_dir = 'poke_dir_samples_last_ckpt'
        else:
            samples_dir = 'poke_dir_samples_best_fvd'

        for sid, x_0, poke, X_tgt, s, poke_coord in zip(start_id, x_0s, randomized_pokes.transpose(0,1), X_tgts, samples, complete_poke_coords):
            x_0 = x_0[None]
            poke = poke
            X_tgt = X_tgt[None]
            poke_coord = poke_coord[:,None]
            samples_list, samples_grid, samples_grid_unlabeled = make_multipoke_grid(x_0, poke, X_tgt, s,
                                                                                    multipoke_coords=poke_coord)



            savedir = os.path.join(self.dirs['generated'], samples_dir, f'sid_{sid}', self.start_time)
            makedirs(savedir, exist_ok=True)

            save_video(samples_grid, os.path.join(savedir, 'overview.mp4'), fps=3)
            save_video(samples_grid_unlabeled, os.path.join(savedir, 'samples_all.mp4'), fps=3)

            for i, sample in enumerate(samples_list):
                if i == 0:
                    savepath = os.path.join(savedir, f'groundtruth.mp4')
                    savepath_en = os.path.join(savedir, f'groundtruth_enrollment.png')
                else:
                    savepath = os.path.join(savedir, f'sample_{i}.mp4')
                    savepath_en = os.path.join(savedir, f'sample_{i}_enrollment.png')

                save_video(sample, savepath, fps=3)
                enrollment_plot = np.concatenate(list(sample), axis=1)
                enrollment_plot = cv2.cvtColor(enrollment_plot, cv2.COLOR_RGB2BGR)
                # make the corresponding enrollment (first look what its like
                cv2.imwrite(savepath_en, enrollment_plot)



    def _generate_samples(self,batch):
        start_id = batch['sample_ids'][:,0].cpu().numpy()
        samples = self.forward_sample(batch,self.n_test_samples,n_logged_vids=self.config['testing']['test_batch_size'],
                                      show_progress=self.config['testing']['verbose'])

        # change dimensions to obtain samppermuteles in right order
        samples = list(torch.stack(samples).transpose(0,1))

        x_0s = batch["images"][:, 0]
        X_tgts = batch["images"]
        pokes = batch["poke"]
        if isinstance(pokes, list):
            poke_coords = pokes[1]
            pokes = pokes[0]
        else:
            poke_coords = None

        if self.config['general']['last_ckpt']:
            samples_dir = 'samples_last_ckpt'
        else:
            samples_dir = 'samples_best_fvd'

        for sid,x_0,poke,X_tgt,s,poke_coord in zip(start_id,x_0s,pokes,X_tgts,samples,poke_coords):

            x_0 = x_0[None]
            poke = poke[None]
            X_tgt = X_tgt[None]
            poke_coord = poke_coord[None]
            samples_list, samples_grid, samples_grid_unlabeled = make_samples_and_samplegrid(x_0,poke,X_tgt,s,
                                                                     poke_coords=poke_coord)



            savedir = os.path.join(self.dirs['generated'],samples_dir,f'sid_{sid}',self.start_time)
            makedirs(savedir,exist_ok=True)

            save_video(samples_grid,os.path.join(savedir,'overview.mp4'),fps=3)
            save_video(samples_grid_unlabeled, os.path.join(savedir, 'samples_all.mp4'), fps=3)

            for i,sample in enumerate(samples_list):
                if i==0:
                    savepath = os.path.join(savedir, f'groundtruth.mp4')
                    savepath_en = os.path.join(savedir, f'groundtruth_enrollment.png')
                else:
                    savepath = os.path.join(savedir,f'sample_{i}.mp4')
                    savepath_en = os.path.join(savedir, f'sample_{i}_enrollment.png')

                save_video(sample,savepath,fps=3)
                enrollment_plot = np.concatenate(list(sample),axis=1)
                enrollment_plot = cv2.cvtColor(enrollment_plot,cv2.COLOR_RGB2BGR)
                # make the corresponding enrollment (first look what its like
                cv2.imwrite(savepath_en,enrollment_plot)

    def _test_transfer(self,batch, batch_id):

        # prepare flow input
        half_poke_size =int(self.trainer.datamodule.dset_val.poke_size / 2)
        X_1 = batch['images']
        X_2, flow, sample_ids2 = batch['nn']
        sample_ids2 = sample_ids2.cpu().numpy()
        sample_ids1 = batch['sample_ids'].cpu().numpy()


        poke1 , poke_coords1 = batch['poke']
        poke_2 = torch.zeros_like(flow)

        # get poke2 which is on exactly the same point in the second image, than in the first one
        for cs,p,fs in zip(poke_coords1,poke_2,flow):
            cs = cs[(cs>=0).all(-1)]
            flow_vals = fs[:,cs[:,0],cs[:,1]].transpose(0,1)
            for f,c in zip(flow_vals,cs):
                p[:,c[0]-half_poke_size:c[0]+half_poke_size+1,c[1]-half_poke_size:c[1]+half_poke_size+1] = f[:,None,None]


        z_1, cond_1 = self.make_flow_input(batch)

        z_2, *_ = self.encode_first_stage(X_2)
        if self.embed_poke_and_image:
            poke_2 = torch.cat([poke_2,X_2[:,0]],dim=1)

        if self.embed_poke_and_image:
            poke1_src2 = torch.cat([poke1, X_2[:, 0]], dim=1)



        poke_emb_2, *_ = self.poke_embedder.encoder(poke_2)
        poke_emb_1, *_ = self.poke_embedder.encoder(poke1_src2)
        # do not sample, as this mapping should be deterministic
        if self.adapt_poke_emb_ssize:
            poke_emb_2 = self.conv_adapt_poke_emb(poke_emb_2)

        if self.use_cond:
            if self.conditioner.be_deterministic:
                cond_2, *_ = self.conditioner.encoder(X_2[:,0])
            else:
                _, cond_2, _ = self.conditioner.encoder(X_2[:,0])


            if self.adapt_cond_ssize:
                cond_2 = self.conv_adapt_cond(cond_2)

        if self.use_cond:
            cond_2 = torch.cat([cond_2, poke_emb_1], dim=1)
        else:
            cond_2 = poke_emb_1


        # get residuals
        r1, _ = self.flow(z_1, cond_1, reverse=False)
        r2, _ = self.flow(z_2, cond_2, reverse=False)

        # transfer
        z_r1_cond2 = self.flow(r1,cond_2, reverse=True)
        #z_r2_cond1 = self.flow(r2, cond_1, reverse=True)
        #sample from residual
        residual_sample = torch.randn_like(r1)
        z_random_cond2 = self.flow(residual_sample,cond_2, reverse=True)

        # decode
        vid_r1_c2 = self.decode_first_stage(z_r1_cond2,X_2)
        vid_random_cond2 = self.decode_first_stage(z_random_cond2,X_2)
        #vid_r2_c1 = self.decode_first_stage(z_r2_cond1, X_1)

        # note that poke_coords are the same in both cases
        transfer_grid, enrollments, single_vids = make_transfer_grids_new(X_1[:,0],X_2[:,0],poke1,X_1,vid_r1_c2,vid_random_cond2,
                                            poke_coords1,make_enrollment=True,sample_ids1=sample_ids1, sample_ids2= sample_ids2)



        savename = path.join(self.transfer_dir,f'transfer_grid-{batch_id}.mp4')
        save_video(transfer_grid,savename,fps=3)

        for e,sid1,sid2, svid in zip(enrollments,sample_ids1,sample_ids2,single_vids):
            savename = path.join(self.transfer_dir,f'transfer_grid-ids_m{sid1[0]}_src{sid2[0]}.png')
            savename_vid = path.join(self.transfer_dir,f'transfer_row-ids_m{sid1[0]}_src{sid2[0]}.mp4')

            e = cv2.cvtColor(e,cv2.COLOR_RGB2BGR)
            cv2.imwrite(savename,e)

            save_video(svid,savename_vid,fps=3)


    def test_step(self, batch, batch_id):
        self.eval()
        with torch.no_grad():
            if self.test_mode == 'fvd':
                self._test_step_fvd(batch)
            elif self.test_mode == 'accuracy':
                self._test_step_metrics(batch,batch_id)
            elif self.test_mode == 'samples':
                self._generate_samples(batch)
            elif self.test_mode == 'kps_acc':
                err_mean_batch = self._test_step_kps_acc(batch,batch_id)
                return err_mean_batch
            elif self.test_mode == 'control_sensitivity':
                self._control_sensitivity(batch,batch_id)
            elif self.test_mode == 'diversity':
                samples = self._test_step_diversity(batch,batch_id)
                return samples
            elif self.test_mode == 'transfer':
                self._test_transfer(batch,batch_id)
            else:
                raise ValueError(f'The specified test_mode is "{self.test_mode}", which is invalid...')


    def test_epoch_end(self, outputs):

        self.console_logger.info(f'******************* TEST SUMMARY on {self.trainer.datamodule.dset_val.__class__.__name__} FOR {self.config["testing"]["n_samples_per_data_point"]} SAMPLES *******************')

        if self.test_mode == 'fvd':

            savedir_vid_samples = path.join(self.dirs['generated'],'fvd_vid_examples')

            makedirs(self.savedir_fvd, exist_ok=True)
            makedirs(savedir_vid_samples, exist_ok=True)

            real_samples = np.stack(self.first_stage_model.fvd_features_true_x0, axis=0)
            fake_samples = np.stack(self.first_stage_model.fvd_features_fake_x0, axis=0)

            self.console_logger.info(f"Generating example videos")
            for i, (r, f) in enumerate(zip(real_samples, fake_samples)):
                savename = path.join(savedir_vid_samples, f"sample{i}.mp4")
                r = np.concatenate([v for v in r], axis=2)
                f = np.concatenate([v for v in f], axis=2)
                all = np.concatenate([r, f], axis=1)

                save_video(all, savename)

                if i >= 4:
                    break

            self.console_logger.info(f"Saving samples to {self.savedir_fvd}")
            np.save(path.join(self.savedir_fvd, "real_samples.npy"), real_samples)
            np.save(path.join(self.savedir_fvd, "fake_samples.npy"), fake_samples)

            self.console_logger.info(f'Finish generation of vid samples.')

        elif self.test_mode == 'accuracy':
            n_pokes = self.trainer.datamodule.dset_val.config['n_pokes']
            lpips_normal = self.lpips_metric.compute().cpu().numpy()
            self.console_logger.info(f'lpips for {n_pokes} pokes is {lpips_normal}')
            ssim_normal = self.ssim.compute().cpu().numpy()
            self.console_logger.info(f'ssim for {n_pokes} pokes is {ssim_normal}')
            # psnr, psnr_dict = self.sample_psnr.compute(n_pokes=n_pokes)
            # self.console_logger.info(f'psnr for {n_pokes} is {psnr}')
            # if len(self.metrics_dict['LPIPS']) == 0:
            #     self.metrics_dict['LPIPS'] = lpips_normal
            #     self.metrics_dict['SSIM'] = lpips_normal
            # else:
            #     self.metrics_dict['SSIM'] = {key: np.append(self.metrics_dict['SSIM'][key], lpips_normal[key]) for key in self.metrics_dict['SSIM']}
            #     self.metrics_dict['LPIPS'] = {key: np.append(self.metrics_dict['LPIPS'][key], lpips_normal[key]) for key in self.metrics_dict['LPIPS']}
            self.metrics_dict['SSIM'] = ssim_normal
            self.metrics_dict['LPIPS'] = lpips_normal

            if 'keypoints_rel' in self.trainer.datamodule.dset_val.datakeys and 'keypoints_abs' in self.trainer.datamodule.dset_val.datakeys:
                kps_dict = self.pose_nn_metric.compute(n_pokes=n_pokes)
                if len(self.metrics_dict['KPS']) == 0:
                    self.metrics_dict['KPS'] = kps_dict
                else:
                    self.metrics_dict['KPS'] = {key: np.append(self.metrics_dict['KPS'][key], kps_dict[key]) for key in self.metrics_dict['KPS']}


            # for convenience
            self.pose_nn_metric.reset()
            self.sample_lpips.reset()
            self.sample_ssim.reset()
            self.ssim.reset()
            self.lpips_metric.reset()

            #return lpips_dict, ssim_dict #,psnr_dict

        elif self.test_mode == 'diversity':
            n_pokes = self.trainer.datamodule.dset_val.config['n_pokes']
            samples = outputs
            exmpls = torch.cat(samples,dim=0)

            self.console_logger.info(f'Range and shape check before divscore calculation: shape is {exmpls.shape}: range is ({exmpls.min()},{exmpls.max()})')

            if self.config['testing']['div_kp']:
                div_score = self.compute_kp_divscore(exmpls,self.device)
            else:
                div_score = compute_div_score(exmpls,self.vggm,device=self.device)

            div_score_mse = compute_div_score_mse(exmpls, device=self.device)
            div_score_lpips = compute_div_score_lpips(exmpls, device=self.device)
            self.div_scores.append(div_score)
            exmpls = exmpls.cpu().numpy()
            savepath = path.join(self.dirs['generated'],'diversity')
            makedirs(savepath, exist_ok=True)

            text = f'Similarity measure_vgg: {div_score}; similarity measure mse: {div_score_mse}; similarity measure lpips: {div_score_lpips}\n'
            self.console_logger.info(text)

            np.save(path.join(savepath,f'samples_diversity_{n_pokes}_pokes.npy'),exmpls)

            #self.console_logger.info(f'Average cosine distance in vgg features space for {n_pokes} pokes: {div_score}')
            divscore_path = path.join(self.metrics_dir,f'divscore.txt')
            with open(divscore_path,'a+') as f:
                f.writelines(text)

            return div_score
        elif self.test_mode == 'kps_acc':

            errs = np.concatenate([o[0] for o in outputs])
            errs_mse = np.concatenate([o[1] for o in outputs])
            mean_err = errs.mean()
            std_err = errs.std()
            mean_mse = errs_mse.mean()
            std_mse = errs_mse.std()
            self.console_logger.info(f'Average targeted keypoint MSE: {mean_err}, std: {std_err} ; torch MSE {mean_mse}, std: {std_mse}')
            kp_err_path = path.join(self.metrics_dir, 'kp_err_targeted.txt')
            with open(kp_err_path, 'a+') as f:
                f.writelines(f'Average targeted keypoint MSE: {mean_err}, std: {std_err} ; torch MSE {mean_mse}, std: {std_mse}\n')

            kp_err_data = pd.DataFrame.from_dict({'err_squared_eucl': errs,'mse': errs_mse})
            kp_err_data.to_csv(path.join(self.metrics_dir,'kp_errs_targeted_acc.csv'))


    def compute_kp_divscore(self,videos,device):
        # from data.flow_dataset import PlantDataset
        n_ex, n_samples, seq_length, c, h, w = videos.shape

        divl = []
        with torch.no_grad():
            for video in tqdm(videos, f'Computing diversity score for {n_ex} examples with {n_samples} samples.'):

                video = video.to(device).reshape(-1, *video.shape[2:])
                kps_raw = self.posenet(video)
                kps_abs, kps_rel = self.posenet.postprocess(kps_raw)

                for j in range(n_samples):
                    for k in range(n_samples):
                        if j != k:
                            f = kps_rel.reshape(n_samples, seq_length, *kps_rel.shape[1:])
                            divl.append(np.linalg.norm((f[j] - f[k]).reshape(-1, 2)).mean())

        divl = np.asarray(divl).mean()

        return divl