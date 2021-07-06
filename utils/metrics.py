import torch
from torch import nn
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import ssim,psnr
from functools import partial
import os
import numpy as np
from scipy import linalg
import torch.nn.functional as F
from torchvision.models import inception_v3
import torchvision
from collections import namedtuple
from tqdm import tqdm
from lpips import LPIPS as lpips_net


from utils.logging import make_nn_var_plot


class metric_vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out

def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=0, keepdim=True))
    return x / (norm_factor + eps)

def normalize_input_vgg(input_tensor):
    from kornia.enhance.normalize import normalize
    # in between [0,1]
    out = (input_tensor + 1.) / 2.
    # vgg normzalization
    out = normalize(out, mean = torch.tensor([0.485, 0.456, 0.406]).type_as(out),
                    std= torch.tensor([0.229, 0.224, 0.225]).type_as(out))

    return out

def compute_div_score(exmpls,feature_extractor,device:torch.device=None):
    n_ex, n_samples, seq_length, c, h, w = exmpls.shape
    d = torch.nn.CosineSimilarity(dim=0)
    # d = torch.nn.CosineSimilarity(dim=1)


    divl = []
    with torch.no_grad():
        for video in tqdm(exmpls, f'Computing VGG-cosine diversity score for {n_ex} examples with {n_samples} samples.'):
            if device is not None:
                video = video.to(device)
            in_tensor = normalize_input_vgg(video.reshape(-1, c, h, w))
            # in_tensor = video.reshape(-1, c, h, w)
            fmap = feature_extractor(in_tensor)
            for j in range(n_samples):
                for k in range(n_samples):
                    if j != k:
                        for l in range(5):
                            f = fmap[l].reshape(n_samples, seq_length, *fmap[l].shape[1:])
                            divl.append(d(normalize_activation(f[j]),
                                          normalize_activation(f[k])).mean().item())
                            # divl.append(d(f[j],f[k]).mean().item())
                            # f = fmap[l].reshape(n_samples, seq_length, -1)
                            # divl.append(d(normalize_activation(f[j]),
                            #               normalize_activation(f[k])).cpu().numpy().mean())


    divl = np.asarray(divl).mean()
    return divl

def compute_div_score_mse(exmpls,device:torch.device=None):
    n_ex, n_samples, seq_length, c, h, w = exmpls.shape
    # d = torch.nn.CosineSimilarity(dim=1)


    divl = []
    with torch.no_grad():
        for video in tqdm(exmpls, f'Computing MSE diversity score for {n_ex} examples with {n_samples} samples.'):
            if device is not None:
                video = video.to(device)
            #in_tensor = normalize_input_vgg(video.reshape(-1, c, h, w))
            # fmap = feature_extractor(in_tensor)
            for j in range(n_samples):
                for k in range(n_samples):
                    if j != k:
                        mse = ((video[j]-video[k])**2).mean().cpu().numpy()
                        divl.append(mse)


    divl = np.asarray(divl).mean()
    return divl


def compute_div_score_lpips(exmpls,device):
    n_ex, n_samples, seq_length, c, h, w = exmpls.shape
    measure = lpips_net().to(device)
    divl = []
    with torch.no_grad():
        for video in tqdm(exmpls, f'Computing LPIPS diversity score for {n_ex} examples with {n_samples} samples.'):
            if device is not None:
                video = video.to(device)
            # in_tensor = normalize_input_vgg(video.reshape(-1, c, h, w))
            # fmap = feature_extractor(in_tensor)
            for j in range(n_samples):
                for k in range(n_samples):
                    if j != k:
                        diff = measure(video[j],video[k]).cpu().numpy()
                        divl.append(diff)

    divl = np.asarray(divl).mean()
    return divl




class SampleMetric(Metric):

    def __init__(self, measure:Metric, logger, n_samples, key:str,reduction=False):
        super().__init__()
        self.n_max_samples = n_samples
        self.logger = logger
        self.reduction = reduction
        msg = 'enabled' if self.reduction else 'disabled'
        self.logger.info(f'set up {key} sample metric. Reduction mode is {msg}.')
        self.add_state('nn_val_per_frame', [], dist_reduce_fx='cat')
        self.add_state('std_per_frame', [], dist_reduce_fx='cat')
        self.add_state('mean_per_frame', [], dist_reduce_fx='cat')
        self.add_state('n_samples', torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('val', torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')

        self.measure = measure
        self.metric_name = key



    def update(self,pred, target):
        """

        :param kps_pred: predicted keypoints of video sequence,  shape: (batch,n_samples_per_batch, n_frames ,channels, height, width)
        :param kps_target: groundtruth keypoints of video sequence shape: (batch, 1 , n_frames ,channels, height, width)
        :return:
        """
        if self.n_samples.int() < self.n_max_samples:

            bs,ns,s,c,h,w = pred.shape
            #all_target = torch.cat([target]*ns,dim=1)
            proc_vals=[]
            # for large amount of data, avoid gpu running out of memory
            for p,t in zip(pred,target):
                t = torch.cat([t] * ns, dim=0)
                if self.reduction:
                    act_val = self.measure(p.reshape(-1, c, h, w).type_as(target), t.reshape(-1, c, h, w)).squeeze()
                else:
                    act_val = self.measure(p.reshape(-1, c, h, w).type_as(target), t.reshape(-1, c, h, w))
                    act_val = act_val.mean(dim=[1,2,3])
                proc_vals.append(act_val.cpu())

            # batch dimension first
            val_all = torch.stack(proc_vals,dim=0)
            val_global_samples = val_all.reshape(bs,ns,s)
            min_ids = torch.argmin(val_global_samples.mean(-1),1)

            min_ids = min_ids[:,None].repeat(1,val_global_samples.size(2))[:,None]
            val_nn_per_frame = val_global_samples.gather(1,min_ids).squeeze(1).cpu()
            std_per_frame = val_global_samples.std(dim=1).cpu()
            mean_per_frame = val_global_samples.mean(dim=1).cpu()
            self.nn_val_per_frame.append(val_nn_per_frame)
            self.std_per_frame.append(std_per_frame)
            self.mean_per_frame.append(mean_per_frame)
            self.n_samples += pred.size(0)
            self.val += val_nn_per_frame.mean(1).sum()

    def compute(self):
        meanval = self.val.float() / self.n_samples

        min_val_per_frame = torch.cat(self.nn_val_per_frame, dim=0).mean(0).numpy()
        mean_per_frame_err = torch.cat(self.mean_per_frame, dim=0).mean(0).numpy()
        avg_std_inside_samples_per_frame = torch.cat(self.std_per_frame, dim=0).mean(0).numpy()
        data_dict_nn = {f'{self.metric_name} NN': min_val_per_frame,
                        f'Mean {self.metric_name} per Frame': mean_per_frame_err,
                        'Std per Frame': avg_std_inside_samples_per_frame
                        }

        return meanval, data_dict_nn

class SampleLPIPS(SampleMetric):

    def __init__(self, logger, n_samples,):
        metric = lpips_net()
        super().__init__(metric,logger,n_samples, key='LPIPS',reduction=True)

    def compute(self,n_pokes=None):
        lpips, data_dict = super().compute()
        data_dict.update({'Time': np.arange(data_dict['LPIPS NN'].shape[0]),})
        if n_pokes is not None:
            data_dict.update({'Number of Pokes': np.full_like(data_dict['LPIPS NN'],n_pokes,dtype=int)})
        return lpips, data_dict

class SamplePSNR(SampleMetric):

    def __init__(self, logger, n_samples,):
        metric = partial(psnr,reduction='none')
        super().__init__(metric,logger,n_samples, key='PSNR',reduction=False)

    def compute(self,n_pokes=None):
        psnr_val, data_dict = super().compute()
        data_dict.update({'Time': np.arange(data_dict['PSNR NN'].shape[0]),})
        if n_pokes is not None:
            data_dict.update({'Number of Pokes': np.full_like(data_dict['PSNR NN'],n_pokes,dtype=int)})
        return psnr_val, data_dict


class SampleSSIM(SampleMetric):

    def __init__(self, logger, n_samples,):
        metric = partial(ssim,reduction='none')
        super().__init__(metric,logger,n_samples, key='SSIM',reduction=False)

    def compute(self,n_pokes=None):
        ssim, data_dict = super().compute()
        data_dict.update({'Time': np.arange(data_dict['SSIM NN'].shape[0]),})
        if n_pokes is not None:
            data_dict.update({'Number of Pokes': np.full_like(data_dict['SSIM NN'],n_pokes,dtype=int)})
        return ssim, data_dict

class KPSMetric(Metric):

    def __init__(self, logger,n_samples=1000, savedir=None):
        """
        Keypoint metric for nearest neighbour and variance
        :param n_samples:
        :param savedir:
        """
        super().__init__(compute_on_step=False,dist_sync_on_step=True)
        self.n_max_samples = n_samples
        self.logger = logger
        self.logger.info('set up keypoints metric')

        self.add_state('nn_err_per_frame', [], dist_reduce_fx='cat')
        self.add_state('std_per_frame', [], dist_reduce_fx='cat')
        self.add_state('mean_per_frame',[],dist_reduce_fx='cat')
        self.add_state('n_samples', torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('nn_err',torch.tensor(0, dtype=torch.float), dist_reduce_fx='sum')

        self.mse = nn.MSELoss(reduction='none')

        if savedir is not None:
            self.savedir = os.path.join(savedir,'kps_stats')
        else:
            self.savedir = savedir


    def update(self,kps_pred, kps_target):
        """

        :param kps_pred: predicted keypoints of video sequence,  shape: (batch,n_samples_per_batch, n_frames ,n_kps, 2)
        :param kps_target: groundtruth keypoints of video sequence shape: (batch,1, n_frames, n_kps, 2 )
        :return:
        """
        if self.n_samples.int() < self.n_max_samples:
            mse = self.mse(kps_pred,kps_target)
            #scompute nearest neighbour sequence in keypoint space
            mse_global = torch.mean(mse,dim=[2,3,4])
            min_ids = torch.argmin(mse_global,dim=1)
            min_ids = min_ids[:,None].repeat(1,mse.size(2))[:,None]
            mse_per_frame = torch.mean(mse,dim=[3,4])
            nns_per_frame = mse_per_frame.gather(1,min_ids).squeeze(1).cpu()
            std_per_frame = mse_per_frame.std(dim=1).cpu()
            mean_per_frame = mse_per_frame.mean(dim=1).cpu()
            self.nn_err_per_frame.append(nns_per_frame)
            self.std_per_frame.append(std_per_frame)
            self.mean_per_frame.append(mean_per_frame)
            self.n_samples += kps_pred.size(0)
            self.nn_err += nns_per_frame.mean(1).sum()

    def compute(self,n_pokes=0):
        nn_err_mean = self.nn_err.float() / self.n_samples

        if self.savedir is None:
            nn_mse_err_per_frame = torch.cat(self.nn_err_per_frame, dim=0).mean(0).numpy()
            mean_per_frame_err = torch.cat(self.mean_per_frame,dim=0).mean(0).numpy()
            data_dict_nn = {'NN MSE': nn_mse_err_per_frame,
                            'Mean MSE per Frame':mean_per_frame_err,
                            'Std per Frame': torch.cat(self.std_per_frame, dim=0).mean(0).numpy(),
                            'Time': np.arange(nn_mse_err_per_frame.shape[0]),
                            'Number of Pokes': np.full_like(nn_mse_err_per_frame,n_pokes,dtype=int)}
            return data_dict_nn
        else:
            self.logger.info(f'Saving plots and related data to "{self.savedir}"')
            mean_err_per_frame=torch.cat(self.nn_err_per_frame,dim=0)
            mean_std_per_frame = torch.cat(self.std_per_frame, dim=0)
            mean_nn_mse_over_samples = mean_err_per_frame.mean(0).numpy()
            mean_std_per_frame_over_samples = mean_std_per_frame.mean(0).numpy()

            make_nn_var_plot(n_pokes,mean_nn_mse_over_samples,
                             std_per_frame=mean_std_per_frame_over_samples,savedir=self.savedir)

            return nn_err_mean



class FVD(Metric):

    def __init__(self,n_samples):
        super().__init__(compute_on_step=False,dist_sync_on_step=True)

        self.n_max_samples = n_samples
        self.i3d = I3D(400, 'rgb')
        self.model_path = 'logs/I3D/i3d_kinetics_rgb.pth'
        if 'DATAPATH' in os.environ:
            self.model_path = os.path.join(os.environ['DATAPATH'], self.model_path[1:])
        # model_path = '/export/home/ablattma/tools/kinetics-i3d-Pytorch/data/pytorch_checkpoints/rgb_imagenet.pkl'
        state_dict = torch.load(self.model_path, map_location="cpu")
        self.i3d.load_state_dict(state_dict)
        self.i3d.eval()

        self.add_state('features_fake', [], dist_reduce_fx='cat')
        self.add_state('features_real', [], dist_reduce_fx='cat')
        self.add_state('n_samples',torch.tensor(0,dtype=torch.int), dist_reduce_fx='sum')

    def load_state_dict(self, state_dict,strict: bool = True):
        st = torch.load(self.model_path, map_location="cpu")
        super().load_state_dict(st,strict)

    def update(self, pred, target,cuda=False) -> None:
        if torch.less(self.n_samples,self.n_max_samples):
            batch_size = pred.size(0)
            data_gen, data_orig = preprocess(pred, target)
            # input is expected to be on matching device
            self.i3d.eval()
            feats_fake  = get_activations(data_gen, self.i3d, batch_size, cuda)
            feats_real = get_activations(data_orig, self.i3d, batch_size, cuda)

            self.features_real.append(feats_real)
            self.features_fake.append(feats_fake)
            self.n_samples += batch_size

    def compute(self):
        feats_real = np.concatenate(self.features_real,axis=0)[:self.n_max_samples]
        feats_fake = np.concatenate(self.features_fake, axis=0)[:self.n_max_samples]

        m_real, s_real = calculate_moments(feats_real)
        m_fake, s_fake = calculate_moments(feats_fake)

        return calculate_frechet_distance(m_fake,s_fake,m_real,s_real)

class FID(Metric):

    def __init__(self,n_samples,normalize_range):
        super().__init__(compute_on_step=False,dist_sync_on_step=True)
        self.n_max_samples = n_samples
        self.inception_model = FIDInceptionModel(normalize_range=normalize_range)
        self.inception_model.eval()

        self.add_state('features_fake', [], dist_reduce_fx='cat')
        self.add_state('features_real', [], dist_reduce_fx='cat')
        self.add_state('n_samples', torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')


    def update(self, pred, target) -> None:
        if torch.less(self.n_samples,self.n_max_samples):
            with torch.no_grad():
                self.features_real.append(self.inception_model(target).cpu().numpy())
                self.features_fake.append(self.inception_model(pred).cpu().numpy())

            self.n_samples += pred.size(0)


    def compute(self):
        return compute_fid(self.features_real,self.features_fake)


def scale_imgs_lpips(x,input_format):
    if input_format == "float-1":
        out = x
    elif input_format == "float0":
        out = x * 2. - 1.
    elif input_format =="uint":
        out = (x / 127.5) - 1.
    else:
        raise ValueError(f'Specified Input Format "{input_format}" is invalid.')

    return out

class LPIPS(Metric):

    def __init__(self, input_format="float-1"):
        super().__init__(dist_sync_on_step=True)

        # self.lpips_func = lpips_net()
        # for param in self.lpips_func.parameters():
        #     param.requires_grad = False

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("lpips_acc", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.scale_fx = partial(scale_imgs_lpips,input_format=input_format)


        # model_path = os.path.abspath(os.path.join(inspect.getfile(self.lpips_func.__init__), '..', 'weights/v%s/%s.pth' % (self.lpips_func.version, self.lpips_func.pnet_type)))
        # self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    def update(self, lpips_net, preds: torch.Tensor, target: torch.Tensor):
        preds = self.scale_fx(preds)
        target = self.scale_fx(target)

        with torch.no_grad():
            lpips_batch = lpips_net(target,preds).cpu()
            self.lpips_acc += torch.sum(lpips_batch)
            n_samples = lpips_batch.numel()
            self.total += n_samples

    def compute(self):
        return self.lpips_acc.float() / self.total


class SSIM_custom(Metric):

    def __init__(self):
        super().__init__(dist_sync_on_step=True)

        self.add_state("total",default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ssim_acc", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self,preds,targets):

        ssim_batch = ssim(preds,targets)
        self.ssim_acc += ssim_batch
        self.total += 1

    def compute(self):
        return self.ssim_acc.float() / self.total

class PSNR_custom(Metric):

    def __init__(self):
        super().__init__(dist_sync_on_step=True)

        self.add_state("total",default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("psnr_acc", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self,preds,targets):

        psnr_batch = psnr(preds,targets)
        self.psnr_acc += psnr_batch
        self.total += 1

    def compute(self):
        return self.psnr_acc.float() / self.total

def mean_cov(features):
    mu = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mu,cov

def compute_fid(real_features, fake_features, eps=1e-6):
    # Taken and adapted from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py

    if not isinstance(real_features,np.ndarray):
        real_features = np.concatenate(real_features,axis=0)
        fake_features = np.concatenate(fake_features,axis=0)

    #filter out nans
    filtered_ids = np.flatnonzero(np.logical_not(np.isnan(fake_features)).any(axis=-1))

    real_features = real_features[filtered_ids]
    fake_features = fake_features[filtered_ids]

    mu1, cov1 = mean_cov(real_features)
    mu2, cov2 = mean_cov(fake_features)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(cov1)
    sigma2 = np.atleast_2d(cov2)

    assert (
            mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
            sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"

        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class FIDInceptionModel(nn.Module):
    def __init__(self, normalize_range=True):
        super().__init__()
        self.v3 = inception_v3(pretrained=True)


        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
                .unsqueeze(dim=0)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
                .unsqueeze(dim=0)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1),
        )

        self.resize = nn.Upsample(size=(299,299),mode="bilinear")
        self.normalize_range = normalize_range

    def forward(self, x):
        x = self.resize(x)
        if self.normalize_range:
            # normalize in between 0 and 1
            x = (x + 1.) / 2.
        else:
            x = x.to(torch.float) / 255.
        # normalize to demanded values
        x = (x - self.mean) / self.std

        # this reimpleents the respective layers of the inception model, see model definition
        for name, submodule in self.v3._modules.items():
            if name == 'AuxLogits':
                continue
            x = submodule(x)
            if name == "Mixed_7c":
                break
            elif name == "Conv2d_4a_3x3" or name == "Conv2d_2b_3x3":
                x = F.avg_pool2d(x, kernel_size=3, stride=2)

        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = torch.flatten(out, 1)

        return out


"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def calculate_frechet_distance_dic(act1, act2, eps=1e-6):
    act1 = np.stack(act1, 0)
    act2 = np.stack(act2, 0)
    mu1, sigma1 = np.mean(act1, 0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, 0), np.cov(act2, rowvar=False)

    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)/3.3

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """



    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def get_activations(data, model, batch_size=50, cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- data        : Tensor of images
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    n_samples = data.size(0)

    if n_samples % batch_size != 0:
        pass
        # print(('Warning: number of images is not a multiple of the '
        #        'batch size. Some samples are going to be ignored.'))
    if batch_size > n_samples:
        # print(('Warning: batch size is bigger than the data size. '
        #        'Setting batch size to data size'))
        batch_size = n_samples

    n_batches = n_samples // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, 400))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)

        start = i * batch_size
        end = start + batch_size

        batch = data[start:end]
        if cuda:
            batch = batch.cuda()
        with torch.no_grad():
            pred = model(batch.permute(0, 2, 1, 3, 4))[1]
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')
    return pred_arr

def calculate_moments(data):
    filtered_ids = np.flatnonzero(np.logical_not(np.isnan(data)).any(axis=-1))
    act = data[filtered_ids]

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_activation_statistics(data, model, batch_size=50, cuda=True, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(data, model, batch_size, cuda, verbose)

    # filter out nans
    filtered_ids = np.flatnonzero(np.logical_not(np.isnan(act)).any(axis=-1))
    act = act[filtered_ids]

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_FVD(model, data_gen, data_orig, batch_size, cuda=True):
    """Calculates the FID for two tensors"""
    data_gen, data_orig = preprocess(data_gen, data_orig)
    m1, s1 = calculate_activation_statistics(data_gen, model, batch_size, cuda)
    m2, s2 = calculate_activation_statistics(data_orig, model, batch_size, cuda)
    FVD = calculate_frechet_distance(m1, s1, m2, s2)

    return FVD

def compute_activations(model, data_gen, data_orig, batch_size, cuda=True):
    data_gen, data_orig = preprocess(data_gen, data_orig)
    return get_activations(data_orig, model, batch_size, cuda), get_activations(data_gen, model, batch_size, cuda)

def preprocess(data_gen, data_orig):

    data_gen = F.interpolate(data_gen.reshape(-1, *data_gen.shape[2:]), mode='bilinear', size=(224, 224),
                             align_corners=True).reshape(*data_gen.shape[:2], 3, 224, 224)
    data_orig = F.interpolate(data_orig.reshape(-1, *data_orig.shape[2:]), mode='bilinear', size=(224, 224),
                              align_corners=True).reshape(*data_orig.shape[:2], 3, 224, 224)

    if data_gen.min() < 0:
        data_gen = denorm(data_gen)

    if data_orig.min() < 0:
        data_orig = denorm(data_orig)

    return data_gen, data_orig

def denorm(x):
    return (x + 1.0)/2.0

def load_model():
    model = I3D(400, 'rgb')
    state_dict = torch.load('/export/home/mdorkenw/code/kinetics_i3d_pytorch/model/model_rgb.pth', map_location="cpu")
    model.load_state_dict(state_dict)
    _ = model.eval()

    return model



def get_padding_shape(filter_shape, stride, mod=0):
    """Fetch a tuple describing the input padding shape.

    NOTES: To replicate "TF SAME" style padding, the padding shape needs to be
    determined at runtime to handle cases when the input dimension is not divisible
    by the stride.
    See https://stackoverflow.com/a/49842071 for explanation of TF SAME padding logic
    """
    def _pad_top_bottom(filter_dim, stride_val, mod):
        if mod:
            pad_along = max(filter_dim - mod, 0)
        else:
            pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for idx, (filter_dim, stride_val) in enumerate(zip(filter_shape, stride)):
        depth_mod = (idx == 0) and mod
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val, depth_mod)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)

    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        self.stride = stride
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
            if stride[0] > 1:
                padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                                  mod in range(stride[0])]
            else:
                padding_shapes = [padding_shape]
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pads = [torch.nn.ConstantPad3d(x, 0) for x in padding_shapes]
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            # This is not strictly the correct map between epsilons in keras and
            # pytorch (which have slightly different definitions of the batch norm
            # forward pass), but it seems to be good enough. The PyTorch formula
            # is described here:
            # https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
            tf_style_eps = 1E-3
            self.batch3d = torch.nn.BatchNorm3d(out_channels, eps=tf_style_eps)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            # Determine the padding to be applied by examining the input shape
            pad_idx = inp.shape[2] % self.stride[0]
            pad_op = self.pads[pad_idx]
            inp = pad_op(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.stride = stride
            if stride[0] > 1:
                padding_shapes = [get_padding_shape(kernel_size, stride, mod) for
                                  mod in range(stride[0])]
            else:
                padding_shapes = [padding_shape]
            self.pads = [torch.nn.ConstantPad3d(x, 0) for x in padding_shapes]
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        pad_idx = inp.shape[2] % self.stride[0]
        pad_op = self.pads[pad_idx]
        inp = pad_op(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(
            in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(
            in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(
            out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(
            in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(
            out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(
            in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out

class I3D(torch.nn.Module):
    def __init__(self,
                 num_classes,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception'):
        super(I3D, self).__init__()

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        elif modality == 'flow':
            in_channels = 2
        else:
            raise ValueError(
                '{} not among known modalities [rgb|flow]'.format(modality))
        self.modality = modality

        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding='SAME')
        # 1st conv-pool
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # conv conv
        conv3d_2b_1x1 = Unit3Dpy(
            out_channels=64,
            in_channels=64,
            kernel_size=(1, 1, 1),
            padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(
            out_channels=192,
            in_channels=64,
            kernel_size=(3, 3, 3),
            padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024,
            out_channels=self.num_classes,
            kernel_size=(1, 1, 1),
            activation=None,
            use_bias=True,
            use_bn=False)
        self.softmax = torch.nn.Softmax(1)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inp):
        # Preprocessing
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        out_logits = out
        out = self.softmax(out_logits)
        return out, out_logits



def compute_fvd(real_videos,fake_videos, device,logger):
    import tensorflow.compat.v1 as tf
    from testing.frechet_video_distance import preprocess,Embedder,calculate_fvd
    from tqdm import tqdm
    # required for fvd computation


    # config = tf.ConfigProto()
    # config.gpu_options.visible_device_list = f"{device}"



    devs = tf.config.experimental.get_visible_devices("GPU")
    target_dev = [d for d in devs if d.name.endswith(str(device))][0]
    tf.config.experimental.set_visible_devices(target_dev, 'GPU')
    logger.info("Compute fvd score.")
    #dev = f"/gpu:{device}"
    logger.info(f"using device {device}")
    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            # construct graph
            sess = tf.Session()

            input_shape = real_videos[0].shape
            input_real = tf.placeholder(dtype=tf.uint8, shape=input_shape)
            input_fake = tf.placeholder(dtype=tf.uint8, shape=input_shape)

            real_pre = preprocess(input_real, (224, 224))

            emb_real = Embedder(real_pre)
            embed_real = emb_real.create_id3_embedding(real_pre)
            fake_pre = preprocess(input_fake, (224, 224))
            emb_fake = Embedder(fake_pre)
            embed_fake = emb_fake.create_id3_embedding(fake_pre)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            real, fake = [], []
            for rv, fv in tqdm(zip(real_videos, fake_videos)):
                # real_batch = ((rv + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy()
                # fake_batch = ((fv + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy()
                # real_batch = ((rv + 1.) * 127.5).cpu().numpy()
                # fake_batch = ((fv + 1.) * 127.5).cpu().numpy()
                feed_dict = {input_real: rv, input_fake: fv}
                r, f = sess.run([embed_fake, embed_real], feed_dict)
                real.append(r)
                fake.append(f)
            print('Compute FVD score')
            real = np.concatenate(real, axis=0)
            fake = np.concatenate(fake, axis=0)
            embed_real = tf.placeholder(dtype=tf.float32, shape=(real.shape[0], 400))
            embed_fake = tf.placeholder(dtype=tf.float32, shape=(real.shape[0], 400))
            result = calculate_fvd(embed_real, embed_fake)
            feed_dict = {embed_real: real, embed_fake: fake}
            fvd_val = sess.run(result, feed_dict)
            sess.close()


            logger.info(f"Results of fvd computation: fvd={fvd_val}")

    # for being sure
    return fvd_val


if __name__ == '__main__':

    seq_gen  = torch.randn((100, 16, 3, 128, 128))
    seq_orig = torch.randn((100, 16, 3, 128, 128))
    batch_size = 20

    ## Load pretrained model for dynamic textures
    device = torch.device('cuda:0')
    model = I3D(400).to(device)
    state_dict = torch.load('/export/data/ablattma/i3d_rgb_imagenet.pt', map_location="cpu")
    model.load_state_dict(state_dict)
    _ = model.eval()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    fid_value = calculate_FVD(model, seq_gen, seq_orig, batch_size, True)
    print('FVD: ', fid_value)