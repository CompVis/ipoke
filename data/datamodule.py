from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,WeightedRandomSampler
import numpy as np
from copy import deepcopy

from data import get_dataset
from data.samplers import FixedLengthSampler

class StaticDataModule(LightningDataModule):

    def __init__(self,config, datakeys,debug=False):
        from data.flow_dataset import IperDataset
        super().__init__()
        self.config = config
        self.datakeys = datakeys
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["n_workers"]
        self.zero_poke = "zero_poke" in self.config and self.config["zero_poke"]
        self.dset, self.transforms = get_dataset(self.config)
        self.dset_train = self.dset(self.transforms,self.datakeys,self.config, train=True, debug=debug)
        if isinstance(self.dset_train,IperDataset) and self.dset_train.yield_videos:
            self.test_datakeys = self.datakeys + ['keypoints_rel','keypoints_abs','keypoint_poke' ,'nn'] #
            self.val_datakeys = self.datakeys + ['keypoints_rel', 'keypoints_abs', 'keypoint_poke']
        else:
            self.test_datakeys = self.val_datakeys = self.datakeys

        if self.config['filter'] != 'all':
            self.test_config = deepcopy(self.config)
        else:
            self.test_config = self.config
        self.dset_val = self.dset(self.transforms, self.val_datakeys, self.test_config, train=False,debug=debug)

        self.dset_test = self.dset_val if not isinstance(self,IperDataset) else self.dset(self.transforms, self.test_datakeys, self.test_config, train=False,debug=debug)
        self.val_obj_weighting = self.config['object_weighting'] if 'object_weighting' in self.config else self.dset_val.obj_weighting
        def w_init_fn(worker_id):
            return np.random.seed(np.random.get_state()[1][0] + worker_id)

        self.winit_fn = w_init_fn


    def train_dataloader(self):
        if self.zero_poke:
            sampler = FixedLengthSampler(self.dset_train,self.batch_size,shuffle=True,
                                         drop_last=True,weighting=self.dset_train.obj_weighting,
                                         zero_poke=self.zero_poke,zero_poke_amount=self.config["zero_poke_amount"])
            return DataLoader(self.dset_train,batch_sampler=sampler,num_workers=self.num_workers,worker_init_fn=self.winit_fn)
        else:
            if self.dset_train.obj_weighting:
                sampler = WeightedRandomSampler(weights=self.dset_train.datadict["weights"], num_samples=self.dset_train.datadict["img_path"].shape[0])
                return DataLoader(self.dset_train, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)
            else:
                return DataLoader(self.dset_train,batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def val_dataloader(self):
        if self.zero_poke:
            sampler = FixedLengthSampler(self.dset_val, self.batch_size, shuffle=True,
                                         drop_last=True, weighting=self.val_obj_weighting,
                                         zero_poke=self.zero_poke, zero_poke_amount=self.config["zero_poke_amount"])
            return DataLoader(self.dset_val, batch_sampler=sampler, num_workers=self.num_workers, worker_init_fn=self.winit_fn)
        else:
            if self.val_obj_weighting:
                sampler = WeightedRandomSampler(weights=self.dset_val.datadict["weights"],num_samples=self.dset_val.datadict["img_path"].shape[0])
                return DataLoader(self.dset_val,batch_size=self.batch_size,num_workers=self.num_workers,sampler=sampler)
            else:
                return DataLoader(self.dset_val,batch_size=self.batch_size, num_workers=self.num_workers,shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.dset_test,batch_size=self.config['test_batch_size'], num_workers=self.num_workers,shuffle=True)
