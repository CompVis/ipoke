import torch
from torch import nn
from kornia.geometry.transform import Resize
from kornia.enhance import Normalize
import yaml
from dotmap import DotMap
import os
import numpy as np

from models.pose_estimator.lib.models.pose_resnet import get_pose_net
from models.pose_estimator.lib.core.inference import get_max_preds

class PoseNetWrapper(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.model_path = 'logs/pose_estimator/pose_resnet_152_256x256.pth'
        fp = os.path.dirname(os.path.realpath(__file__))
        configpath = os.path.abspath(os.path.join(fp, "../config/posenet.yaml"))
        with open(configpath,'r') as f:
            cfg = yaml.load(f,Loader=yaml.FullLoader)

        self.cfg = DotMap(cfg)

        #self.cfg = '../config/posenet.yaml'
        self.config = config
        self.input_size = self.config['data']['spatial_size']
        self.resize = Resize((256,256))
        self.normalize = Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))

        self.posenet = get_pose_net(self.cfg,is_train=False)
        self.posenet.load_state_dict(torch.load(self.model_path,map_location='cpu'),strict=False)

    def prepocess_image(self,x):
        out = self.resize(x)
        out = (out + 1.) / 2
        out = self.normalize(out)

        return out

    def forward(self,x):
        self.posenet.eval()
        x = self.prepocess_image(x)
        out = self.posenet(x)
        if isinstance(out,list):
            out = out[-1]

        return out

    def postprocess(self,x):
        if not isinstance(x,np.ndarray):
            x = x.detach().cpu().numpy()
        out, _ = get_max_preds(x)

        #resize abs kps to input spatial size
        out_abs = out * (self.input_size[0] / 64)
        out_rel = out / 64

        return out_abs, out_rel





if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import numpy as np
    from os import path
    from tqdm import tqdm
    from os import makedirs
    import cv2

    from data import get_dataset
    from data.samplers import FixedLengthSampler
    from models.pose_estimator.tools.infer import save_batch_image_with_joints
    from utils.metrics import KPSMetric
    from utils.general import get_logger_old
    from utils.logging import put_text_to_video_row

    # load config
    fpath = path.dirname(path.realpath(__file__))
    logger = get_logger_old(fpath)
    configpath = path.abspath(path.join(fpath, "../config/test_config.yaml"))
    with open(configpath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["fix_seed"]:
        seed = 42
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(42)
        rng = np.random.RandomState(42)

    dset, transforms = get_dataset(config["data"])

    datakeys = ['images','keypoints_rel','keypoints_abs', 'sample_ids']

    test_dataset = dset(transforms, datakeys, config["data"],train=False)

    def init_fn(worker_id):
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


    sampler = FixedLengthSampler(test_dataset, config['data']['batch_size'], shuffle=True,
                                 drop_last=True, weighting=test_dataset.obj_weighting,
                                 zero_poke=config['data']['zero_poke'], zero_poke_amount=config['data']["zero_poke_amount"])
    loader = DataLoader(test_dataset, batch_sampler=sampler, num_workers=config['data']['n_workers'], worker_init_fn=init_fn)

    dev = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')

    model = PoseNetWrapper(config)
    model.to(dev)

    save_dir = f"../data/test_data/{test_dataset.__class__.__name__}"
    save_dir = path.abspath(path.join(fpath,save_dir))

    n_expls_metric = config['n_exmpls_pose_metric']

    kps_metric = KPSMetric(logger,n_samples=n_expls_metric,savedir=save_dir)
    kps_metric.to(dev)

    for id, batch in enumerate(tqdm(loader)):
        if id > n_expls_metric:
            break
        imgs = batch['images'].to(dev)
        kps_abs = batch['keypoints_abs']
        kps_rel = batch['keypoints_rel'].to(dev)

        original_shape = imgs.shape
        if imgs.ndim == 5:
            imgs = imgs.reshape(-1,*imgs.shape[2:])


        with torch.no_grad():
            out_raw = model(imgs)
            pred_abs, pred_rel = model.postprocess(out_raw)


        imgs_with_gt = save_batch_image_with_joints(imgs,kps_abs.reshape(-1,*kps_abs.shape[2:]),[],None,nrow=1,return_image=True)
        imgs_with_pred = save_batch_image_with_joints(imgs,pred_abs,[],None,nrow=1,return_image=True)

        grid = np.concatenate([imgs_with_gt,imgs_with_pred],axis=0)
        grid = cv2.cvtColor(grid,cv2.COLOR_RGB2BGR)
        cv2.imwrite(path.join(save_dir,f'pose_exmpl_{id}.png'),grid)


        # restore time axis
        pred_abs = pred_abs.reshape(*original_shape[:2],*pred_abs.shape[1:])
        pred_rel = torch.from_numpy(pred_rel.reshape(*original_shape[:2], *pred_rel.shape[1:])).to(dev)

        kps_metric.update(pred_rel[:,None],kps_rel[:,None])


    mean_nn_kps = kps_metric.compute()
    logger.info(f'mean nn kps is {mean_nn_kps}')



