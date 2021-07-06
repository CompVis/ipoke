from os import path
import numpy as np
import pickle
from copy import deepcopy
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms as tt
from tqdm import tqdm
import cv2
from PIL import Image
from natsort import natsorted
import os
from glob import glob
from collections import namedtuple

from utils.general import parallel_data_prefetch, LoggingParent,preprocess_image
from data.base_dataset import BaseDataset



class PlantDataset(BaseDataset):
    def __init__(self, transforms, datakeys, config, train=True, debug=False):

        super().__init__(transforms, datakeys, config,train=train)
        self.logger.info(f"Initializing {self.__class__.__name__}.")
        self.debug = debug

        # set instace specific fixed values which shall not be parameters from yaml
        self._set_instance_specific_values()

        if "DATAPATH" in os.environ:
            self.datapath = path.join(os.environ["DATAPATH"],self.datapath[1:])

        self.subsample_step = config["subsample_step"] if "subsample_step" in config else self.subsample_step

        self.logger.info(f'Subsample step of {self.__class__.__name__} is {self.subsample_step}.')

        filt_msg = "enabled" if self.filter_flow else "disabled"
        self.logger.info(f"Flow filtering is {filt_msg} in {self.__class__.__name__}!")
        self.logger.info(f"Valid lag of {self.__class__.__name__} is {self.valid_lags[0]}")

        # load data
        metafile_path = path.join(self.datapath, f"{self.metafilename}.p")

        self.get_kp_nn = 'get_kp_nn' in self.config and self.config['get_kp_nn'] and not self.train


        with open(metafile_path, "rb") as handle:
            self.data = pickle.load(handle)

        if self.debug:
            self.logger.info('Loading small debug dataset to speedup loading')
            self.data = {key:self.data[key][:5000] for key in self.data}



        # if path.isfile(path.join(self.datapath,"dataset_stats.p")) and self.normalize_flows:
            # with open(path.join(self.datapath,"dataset_stats.p"),"rb") as norm_file:
            #     self.flow_norms = pickle.load(norm_file)

        # choose filter procedure
        available_frame_nrs = np.asarray([int(p.split("/")[-1].split(".")[0].split("_")[-1]) - int(p.split("/")[-1].split(".")[0].split("_")[-2]) for p in self.data["flow_paths"][0]])
        # filter invalid flow_paths
        self.data["flow_paths"] = [p for p in self.data["flow_paths"] if len(p) == len(available_frame_nrs)]

        self.filter_proc = self.config["filter"] if "filter" in self.config else "all"
        # remove invalid video
        # valid_ids = np.logical_not(np.char.startswith(self.data["img_path"],"VID_0_3_1024x1024"))

        # set flow paths in right order after reading in the data
        if "max_fid" not in self.data:
            self.data["flow_paths"] = [natsorted(d) for d in self.data["flow_paths"]]

        # make absolute image and flow paths
        # if self.datapath != "":
        #     self.data["img_path"] = [
        #         path.join(self.datapath, p if not p.startswith("/") else p[1:]) for p in self.data["img_path"]
        #     ]
        #     self.data["flow_paths"] = [
        #         [path.join(self.datapath, f if not f.startswith("/") else f[1:]) for f in fs]
        #         for fs in self.data["flow_paths"]
        #     ]

        # convert to numpy array
        self.data = {key: np.asarray(self.data[key]) for key in self.data}

        # if max fid is not predefined, the videos, the dataset consists of are sufficiently long, such that it doesn't make much of a difference,
        # if some frames at the end are skipped, therefore, we set the last valid fid (which is indicated by "max_fid") to the maximum fid
        # in the respective sequence
        # note that, for shareddataset, there is no need for a max_fid since the model is trained on static images
        if "max_fid" not in self.data:


            available_frame_nrs = np.asarray([int(p.split("/")[-1].split(".")[0].split("_")[-1]) - int(p.split("/")[-1].split(".")[0].split("_")[-2]) for p in self.data["flow_paths"][0]])
            self.data.update({"max_fid": np.zeros((np.asarray(self.data["fid"]).shape[0],max(len(available_frame_nrs),self.valid_lags[0]+1)),dtype=np.int)})
            for vid in np.unique(self.data["vid"]):
                self.data["max_fid"][self.data["vid"] == vid] = np.amax(self.data["fid"][self.data["vid"] == vid])

        # if not self.var_sequence_length and not self.normalize_flows or self.normalize_and_fixed_length:
        # reset valid_lags, such that always the right flow which corresponds to the respective sequence length, is chosen
        if not self.__class__.__name__ == "Human36mDataset":
            available_frame_nrs = np.asarray([int(p.split("/")[-1].split(".")[0].split("_")[-1]) - int(p.split("/")[-1].split(".")[0].split("_")[-2])  for p in self.data["flow_paths"][0]])
            if "n_ref_frames" not in self.config:
                assert self.max_frames * self.subsample_step in available_frame_nrs
                right_lag = int(np.argwhere(available_frame_nrs == self.max_frames * self.subsample_step))
                self.logger.info(f'Last frames of sequence serves as target frame.')
            else:
                self.logger.info(f'Number of frames in between target and start frames is {self.config["n_ref_frames"]}')
                assert self.config["n_ref_frames"]*self.subsample_step in available_frame_nrs
                right_lag = int(np.argwhere(available_frame_nrs==self.config["n_ref_frames"] * self.subsample_step))
            self.valid_lags = [right_lag]

        else:
            assert self.max_frames == 10
            assert self.subsample_step in [1,2]
            self.valid_lags = [0] if self.subsample_step == 1 else [1]

        self.logger.info(f"Dataset is run in fixed length mode, valid lags are {self.valid_lags}.")


        filt_msg = "enabled" if self.filter_flow else "disabled"
        self.logger.info(f"Flow filtering is {filt_msg} in {self.__class__.__name__}!")
        self.logger.info(f"Valid lag of {self.__class__.__name__} is {self.valid_lags[0]}")

        filt_msg = "enabled" if self.obj_weighting else "disabled"
        self.logger.info(f"Object weighting is {filt_msg} in {self.__class__.__name__}!")
        # filt_msg = "enabled" if self.flow_weights else "disabled"
        # self.logger.info(f"Patch weighting is {filt_msg} in {self.__class__.__name__}!")
        filt_msg = "enabled" if self.use_flow_for_weights else "disabled"
        self.logger.info(f"Flow patch extraction is {filt_msg} in {self.__class__.__name__}!")

        if self.filter_proc == "action":
            self.data = {key:self.data[key][self.data["action_id"]==2] for key in self.data}
            self.logger.info('Only considering complex motion in dataset.')
        elif self.filter_proc == "pose":
            self.data = {key: self.data[key][self.data["action_id"] == 1] for key in self.data}
            self.logger.info('Only considering rotating motion in dataset.')

        self.split = self.config["split"]
        split_data, train_indices, test_indices = self._make_split(self.data)

        self.train_indices = train_indices
        self.test_indices = test_indices


        self.datadict = (
            split_data["train"] if self.train else split_data["test"]
        )
        msg = "train" if self.train else "test"

        vids, start_ids = np.unique(self.datadict["vid"],return_index=True)

        # get start and end ids per sequence
        self.eids_per_seq = {vid: np.amax(np.flatnonzero(self.datadict["vid"] == vid)) for vid in vids}
        seids = np.asarray([self.eids_per_seq[self.datadict["vid"][i]] for i in range(self.datadict["img_path"].shape[0])],dtype=np.int)
        self.datadict.update({"seq_end_id": seids})

        self.sids_per_seq = {vid:i for vid,i in zip(vids,start_ids)}

        self.seq_len_T_chunk = {l: c for l,c in enumerate(np.linspace(0,self.flow_cutoff,self.max_frames,endpoint=False))}
        # add last chunk
        self.seq_len_T_chunk.update({self.max_frames: self.flow_cutoff})
        # if self.var_sequence_length:
        #     if "flow_range" in self.datadict.keys():
        #         self.ids_per_seq_len = {length: np.flatnonzero(np.logical_and(np.logical_and(self.datadict["flow_range"][:,1]>self.seq_len_T_chunk[length],
        #                                                                                      np.less_equal(np.arange(self.datadict["img_path"].shape[0]) +
        #                                                                                                    (self.min_frames + length)*self.subsample_step + 1,
        #                                                                                                     self.datadict["seq_end_id"])),
        #                                                                       np.less_equal(self.datadict["fid"],self.datadict["max_fid"][:,self.valid_lags[0]])))
        #                             for length in np.arange(self.max_frames)}
        #     else:
        #         self.ids_per_seq_len = {length: np.flatnonzero(np.less_equal(self.datadict["fid"],self.datadict["max_fid"][:,self.valid_lags[0]])) for length in np.arange(self.max_frames)}


        for length in self.ids_per_seq_len:
            actual_ids = self.ids_per_seq_len[length]
            oids, counts_per_obj = np.unique(self.datadict["object_id"][actual_ids],return_counts=True)
            weights = np.zeros_like(actual_ids,dtype=np.float)
            for oid,c in zip(oids,counts_per_obj):
                weights[self.datadict["object_id"][actual_ids]==oid] = 1. / (c * oids.shape[0])

            self.object_weights_per_seq_len.update({length:weights})


        # if not isinstance(self,IperDataset):
        if self.obj_weighting:
            obj_ids, obj_counts = np.unique(self.datadict["object_id"], return_counts=True)
            weights = np.zeros_like(self.datadict["object_id"], dtype=np.float)
            for (oid, c) in zip(obj_ids, obj_counts):
                weights[self.datadict["object_id"] == oid] = 1. / c

            weights = weights / np.sum(weights)

            self.datadict.update({"weights": weights})
        # else:
        #     weights = self.datadict['weights']
        #     self.datadict['weights'] = weights / np.sum(weights)

        # flow_msg = "Flow normalization enabled!" if self.normalize_flows else "Flow normalization disabled!"

        self.logger.info(
            f'Initialized {self.__class__.__name__} in "{msg}"-mode. Dataset consists of {self.__len__()} samples. '
        )

    def _set_instance_specific_values(self):
        # set flow cutoff to 0.2 as this seems to be a good heuristic for Plants

        self.valid_lags = [1]
        self.flow_cutoff = 0.4
        self.extended_annotations = False
        self.subsample_step = 2
        self.min_frames = 5
        self.obj_weighting = True

        self.metafilename = "meta"
        self.datapath = "data/PlantDataset"

    def _read_flows(self,data):
        read_flows = []
        flow_paths = data
        def proc_flow(flow):
            org_shape = float(flow.shape[-1])
            dsize = None
            if "spatial_size" in self.config:
                dsize = self.config["spatial_size"]
            elif "resize_factor" in self.config:
                dsize = (
                    int(float(flow.shape[1]) / self.config["resize_factor"]),
                    int(float(flow.shape[2]) / self.config["resize_factor"]),
                )

            flow = F.interpolate(
                torch.from_numpy(flow).unsqueeze(0), size=dsize, mode="bilinear", align_corners=True
            ).numpy()

            flow = flow / (org_shape / dsize[0])

            return flow


        for i, flow_path in enumerate(tqdm(flow_paths)):
            try:
                f = np.load(flow_path)
                f = proc_flow(f)
            except ValueError:
                try:
                    f = np.load(flow_path, allow_pickle=True)
                    f = proc_flow(f)
                except Exception as ex:
                    self.logger.error(ex)
                    read_flows.append("None")
                    continue
            except:
                self.logger.error("Fallback error ocurred. Append None and continue")
                read_flows.append("None")
                continue

            read_flows.append(f)

        return np.concatenate(read_flows,axis=0)

    def _read_imgs(self,imgs):
        read_imgs = []

        for img_path in tqdm(imgs):
            img = cv2.imread(img_path)
            # image is read in BGR
            img = preprocess_image(img, swap_channels=True)
            img = cv2.resize(
                img, self.config["spatial_size"], cv2.INTER_LINEAR
            )
            read_imgs.append(img)

        return read_imgs

    def _make_split(self,data):

        vids = np.unique(self.data["vid"])
        split_data = {"train": {}, "test": {}}

        if self.split == "videos":
            # split such that some videos are held back for testing
            self.logger.info("Splitting data after videos")
            shuffled_vids = deepcopy(vids)
            np.random.shuffle(shuffled_vids)
            train_vids = shuffled_vids[: int(0.8 * shuffled_vids.shape[0])]
            train_indices = np.nonzero(np.isin(data["vid"], train_vids))[0]
            test_indices = np.nonzero(np.logical_not(train_indices))[0]
            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

        else:
            self.logger.info(f"splitting data across_videos")
            train_indices = np.asarray([],dtype=np.int)
            test_indices = np.asarray([], dtype=np.int)
            for vid in vids:
                indices = np.nonzero(data["vid"] == vid)[0]
                # indices = np.arange(len(tdata["img_path"]))
                # np.random.shuffle(indices)
                train_indices = np.append(train_indices,indices[: int(0.8 * indices.shape[0])])
                test_indices = np.append(test_indices,indices[int(0.8 * indices.shape[0]) :])


            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

        return split_data, train_indices, test_indices


    def __len__(self):
        return self.datadict["img_path"].shape[0]


class VegetationDataset(PlantDataset):

    def _set_instance_specific_values(self):
        self.filter_flow = False
        self.valid_lags = [0]
        self.flow_cutoff = .3
        self.min_frames = 5
        self.subsample_step = 2
        self.datapath = "/export/data/ablattma/Datasets/vegetation_new/"
        self.metafilename = "vegetation_new"
        self.datadict.update({"train": []})
        self.obj_weighting = True
        # set flow_weights to false
        # self.flow_weights = False

    def _make_split(self,data):
        split_data = {"train":{},"test":{}}
        train_ids = np.flatnonzero(data["train"])
        test_ids = np.flatnonzero(np.logical_not(data["train"]))
        assert np.intersect1d(train_ids,test_ids).size == 0
        split_data["train"] = {
                key: data[key][train_ids] for key in data
            }
        split_data["test"] = {
            key: data[key][test_ids] for key in data
        }

        return split_data, train_ids, test_ids



class TaichiDataset(VegetationDataset):

    def _set_instance_specific_values(self):
        self.filter_flow = True
        self.valid_lags = [1]
        self.flow_cutoff = .1
        self.min_frames = 5
        self.subsample_step = 2
        # self.datapath = "/export/scratch/compvis/datasets/taichi/taichi/"
        self.datapath = 'data/TaichiDataset'
        self.metafilename = "meta"
        self.datadict.update({"train": []})
        self.obj_weighting = False
        # set flow_weights to false
        # self.flow_weights = self.config["flow_weights"] if "flow_weights" in self.config else True
        self.flow_width_factor = 5
        self.target_lags = [10,20]

class IperDataset(PlantDataset):

    def _set_instance_specific_values(self):
        self.filter_flow = self.config["filter_flow"] if "filter_flow"  in self.config else True
        self.flow_width_factor = 5
        self.valid_lags = [0]
        # set flow cutoff to 0.45 as this seems to be a good heuristic for Iper
        self.flow_cutoff = 0.6

        self.bone_ids ={'r_upperarm': (11,12),
                        'r_forearm': (10,11),
                        'l_upperam': (13,14),
                        "l_forearm":(14,15),
                        'spine':(6,7),
                        'l_thigh':(1,2),
                        'r_thigh':(3,4),
                        'r_lowerleg':(0,1),
                        'l_lowerleg':(4,5)}


        self.min_frames = 5

        # if self.config["spatial_size"][0] <= 256:
            # self.datapath = "/export/scratch/compvis/datasets/iPER/processed_256_resized/"
        self.datapath = 'data/IperDataset'
        if path.exists(path.join(self.datapath,"meta_kp_nn.p")):
            self.metafilename = "meta_kp_nn"
            self.logger.info('Loading meta data with keypoints and nearest neighbors.')
        else:
            self.metafilename = "meta_kp"
            self.logger.info('Loading meta data with keypoints.')
            #self.metafilename = "meta_frange_with_keypoints_weights_"
        # else:
        #     self.datapath = "/export/scratch/compvis/datasets/iPER/processed/"
        #     self.metafilename = "iper_full"
        self.datadict.update({"actor_id": [], "action_id": []})

        # set object weighting always to false
        self.obj_weighting = False
        # self.flow_weights = self.config["flow_weights"] if "flow_weights" in self.config else True
        self.use_flow_for_weights = False


    def _make_split(self,data):
        split_data = {"train": {}, "test": {}}

        if self.split == "videos":
            key = "vid"
        elif self.split == "objects":
            key = "object_id"
        elif self.split == "actions":
            key = "action_id"
        elif self.split == "actors":
            key = "actor_id"
        elif self.split == "official":
            # this is the official train test split as in the original paper
            with open(path.join(self.datapath,"train.txt"),"r") as f:
                train_names = f.readlines()

            train_indices = np.asarray([],dtype=np.int)
            for n in train_names:
                n = n.replace("/","_").rstrip()
                train_indices = np.append(train_indices,np.flatnonzero(np.char.find(data["img_path"],n) != -1))

            train_indices = np.sort(train_indices)
            test_indices = np.flatnonzero(np.logical_not(np.isin(np.arange(data["img_path"].shape[0]),train_indices)))

            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

            return split_data, train_indices, test_indices

        else:

            vids = np.unique(self.data["vid"])

            self.logger.info(f"splitting data across_videos")
            train_indices = np.asarray([], dtype=np.int)
            test_indices = np.asarray([], dtype=np.int)
            for vid in vids:
                indices = np.nonzero(data["vid"] == vid)[0]
                # indices = np.arange(len(tdata["img_path"]))
                # np.random.shuffle(indices)
                train_indices = np.append(train_indices, indices[: int(0.8 * indices.shape[0])])
                test_indices = np.append(test_indices, indices[int(0.8 * indices.shape[0]):])

            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

            return split_data, train_indices, test_indices

        # split such that some objects are held back for testing
        self.logger.info(f"Splitting data after {key}")
        ids = np.unique(data[key])
        shuffled_ids = deepcopy(ids)
        np.random.shuffle(shuffled_ids)
        train_ids = shuffled_ids[: int(0.8 * shuffled_ids.shape[0])]
        train_indices = np.flatnonzero(np.isin(data[key], train_ids))
        test_indices = np.flatnonzero(np.logical_not(np.isin(np.arange(self.data["img_path"].shape[0]),train_indices)))

        train_indices = np.sort(train_indices)
        test_indices = np.sort(test_indices)

        split_data["train"] = {
            key: data[key][train_indices] for key in data
        }
        split_data["test"] = {
            key: data[key][test_indices] for key in data
        }


        return split_data, train_indices, test_indices

    def _get_keypoints(self,ids,abs=False,**kwargs):
        kps_list = []
        yield_ids = self._get_yield_ids(ids)

        key = 'keypoints_abs' if abs else 'keypoints_rel'
        for idx in yield_ids:
            kps = self.datadict[key][idx]
            if abs:
                kps = kps  / (256 / self.config['spatial_size'][0])
            kps_list.append(torch.from_numpy(kps))

        kps_out = torch.stack(kps_list).squeeze(0)
        return kps_out

    def _get_nn(self,ids, sample_idx, transforms=None, sample=False, use_fb_aug=False,**kwargs):

        nn_idx = self.datadict['nn_ids'][ids[0]]

        nn_ids = (nn_idx,ids[-1])
        flow = self._get_flow(nn_ids)
        yield_ids = self._get_yield_ids(nn_ids)
        imgs = []

        for i, idx in enumerate(yield_ids):
            faug = use_fb_aug and (i == 0 or i == len(yield_ids) - 1)

            img_path = self.datadict["img_path"][idx]
            img = cv2.imread(img_path)
            img = preprocess_image(img, swap_channels=True)
            # image is read in BGR
            if self.use_lanczos and self.config["spatial_size"] == 64:
                img = np.array(Image.fromarray(img).resize(self.config["spatial_size"], resample=Image.LANCZOS))
            else:
                img = cv2.resize(
                    img, self.config["spatial_size"], cv2.INTER_LINEAR
                )

            # transformations
            img = self.pre_T(img)
            if transforms is not None:
                for t in transforms:
                    img = t(img)
                if faug:
                    bts = self._get_color_transforms()
                    img_back = img
                    for bt in bts:
                        img_back = bt(img_back)
                    img_back = self.post_T(img_back)
            else:
                if self.color_transfs is not None:
                    for t in self.color_transfs:
                        img = t(img)

                if self.geom_transfs is not None:
                    for t in self.geom_transfs:
                        img = t(img)

            img = self.post_T(img)
            if faug:
                img = torch.where(torch.from_numpy(self.mask["img_start"]).unsqueeze(0), img, img_back)
            imgs.append(img)



        return torch.stack(imgs, dim=0).squeeze(dim=0), flow, torch.tensor(list(yield_ids),dtype=torch.int)


class Human36mDataset(PlantDataset):
    def _set_instance_specific_values(self):
        self.valid_lags = [self.config['valid_lags']] if 'valid_lags' in self.config and self.config['normalize_flows'] else [1]
        self.flow_cutoff = 0.3

        self.min_frames = 5
        self.subsample_step = 2


        # self.datapath = "/export/scratch/compvis/datasets/human3.6M/video_prediction"
        self.datapath = "data/Human36mDataset"
        self.metafilename = "meta"
        self.datadict.update({"actor_id": [], "action_id": [], "train": []})

        # set object weighting always to false
        self.obj_weighting = False
        self.filter_flow = False
        self.flow_width_factor = 5
        # self.flow_weights = False
        self.use_flow_for_weights = True
        self.use_lanczos = True




    def _make_split(self,data):

        split_data = {"train": {}, "test": {}}

        if self.split == "official":
            train_ids = np.flatnonzero(data["train"])
            test_ids = np.flatnonzero(np.logical_not(data["train"]))
            assert np.intersect1d(train_ids, test_ids).size == 0
            split_data["train"] = {
                key: data[key][train_ids] for key in data
            }
            split_data["test"] = {
                key: data[key][test_ids] for key in data
            }

            return split_data, train_ids, test_ids
        elif self.split == "gui":
            vids = np.unique(self.data["vid"])

            self.logger.info(f"splitting data across_videos")
            train_indices = np.asarray([], dtype=np.int)
            test_indices = np.asarray([], dtype=np.int)
            for vid in vids:
                indices = np.nonzero(data["vid"] == vid)[0]
                # indices = np.arange(len(tdata["img_path"]))
                # np.random.shuffle(indices)
                train_indices = np.append(train_indices, indices[: int(0.8 * indices.shape[0])])
                test_indices = np.append(test_indices, indices[int(0.8 * indices.shape[0]):])

            split_data["train"] = {
                key: data[key][train_indices] for key in data
            }
            split_data["test"] = {
                key: data[key][test_indices] for key in data
            }

            return split_data, train_indices, test_indices
        else:
            raise ValueError(f'Specified split type "{self.split}" is not valid for Human36mDataset.')




def get_nn(ids,dataset, save_dir=None, visualize = False):

    msg_train = 'train' if dataset.train else 'test'

    # compute nearest neighbours of each image frame based on the esimated keypoints
    if 'keypoints_rel' in dataset.datadict.keys():
        msg = 'posture'
        def measure(idx):
            vid = dataset.datadict['vid'][idx]
            kps = dataset.datadict['keypoints_rel']
            sorted_ids = np.argsort(np.linalg.norm(kps[idx][None] - kps, axis=-1).sum(-1))
            indicator_arr = dataset.datadict['vid'][sorted_ids] != vid
            nearest_with_other_video = np.flatnonzero(indicator_arr)[0]
            nn_sa = sorted_ids[nearest_with_other_video]
            nn_gen= sorted_ids[1]
            # nn_gen = np.argpartition(np.linalg.norm(kps[idx][None] - kps, axis=-1).sum(-1),2)[1]
            # nn_sa = np.sort(np.linalg.norm(kps[idx][None] - kps[dataset.datadict['object_id'] == oid], axis=-1).sum(-1),2)[1]

            return nn_gen, nn_sa
    else:
        measure_loader = DataLoader(dataset,batch_size=256,shuffle=False)

        msg = 'lpips'
        gpu = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available else 'cpu')
        lpips = LPIPS().to(gpu)

        def measure(idx):
            raise NotImplementedError()

    nn_other_vid = np.zeros((ids.shape[0]),dtype=int)
    nn_general = np.zeros((ids.shape[0]), dtype=int)

    print(f'STart NN computation for {nn_general.shape[0]} datapoints ')

    for c,i in enumerate(tqdm(ids,desc=f'Computing nearest neighbours in {msg} space for {msg_train}-set of {dataset.__class__.__name__}')):


        nn_gen, nn_ov = measure(i)

        nn_general[c] = nn_gen
        nn_other_vid[c] = nn_ov

    print(f'Finished nearest neighbour computation')

    if visualize:
        assert save_dir is not None

        # loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=True)

        # visualize nns
        #logger.info('finished computation of nns....visualization of examples')
        for c,idx in enumerate(ids):
            # general nearest neighbour
            query = dataset.datadict['img_path'][idx]
            query = cv2.imread(query)
            query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)


            # general nearest neighbour
            nn_path_gen = dataset.datadict['img_path'][nn_general[c]]
            img_nn_gen = cv2.imread(nn_path_gen)
            img_nn_gen = cv2.cvtColor(img_nn_gen,cv2.COLOR_BGR2RGB)
            # imgs_nn_gen.append(img_nn_gen)

#                   # nearest neighbour with same appearance
            nn_path_sa = dataset.datadict['img_path'][nn_other_vid[c]]
            img_nn_sa = cv2.imread(nn_path_sa)
            img_nn_sa = cv2.cvtColor(img_nn_sa, cv2.COLOR_BGR2RGB)
            # imgs_nn_sa.append(img_nn_sa)

            # imgs_nn_gen = np.concatenate(imgs_nn_gen,axis=2)
            # imgs_nn_sa = np.concatenate(imgs_nn_sa,axis=2)
            #
            # imgs_nn_gen = put_text_to_video_row([imgs_nn_gen],'General NN')[0]
            # imgs_nn_sa = put_text_to_video_row([imgs_nn_sa], 'NN same appearance')[0]
            # query = put_text_to_video_row([query], 'Query')[0]

            grid = np.concatenate([query,img_nn_gen,img_nn_sa],axis=1)

            grid = cv2.cvtColor(grid,cv2.COLOR_RGB2BGR)

            gridsavepath = path.join(save_dir,f'nn_exmpl-{c}.png')
            cv2.imwrite(gridsavepath,grid)


    return nn_other_vid

if __name__ == "__main__":
    import yaml
    import torch
    from torchvision import transforms as tt
    from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
    import cv2
    from os import makedirs
    from tqdm import tqdm
    from lpips import LPIPS
    from functools import partial


    from data import get_dataset
    from data.samplers import FixedLengthSampler
    from utils.logging import make_flow_video_with_samples
    from utils.logging import make_flow_img_grid, put_text_to_video_row, vis_kps
    from utils.general import  parallel_data_prefetch

    # load config
    fpath = path.dirname(path.realpath(__file__))
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


    #logger = get_logger_old(fpath)

    make_overlay = config["general"]["overlay"] if "overlay" in config["general"] else False

    postfix = "poke_coords" if config["poke_coords"] else "poke"

    # generate dataset
    dset, transforms = get_dataset(config["data"])

    only_images = False
    datakeys = ["images","flow", "poke", "original_flow", 'sample_ids']



    compute_nn = config['nn_computation']

    if compute_nn:

        def init_fn(worker_id):
            return np.random.seed(np.random.get_state()[1][0] + worker_id)

        test_dataset = dset(transforms, datakeys, config["data"], train=False)
        train_dataset = dset(transforms, datakeys, config["data"], train=True)
        save_dir = f"test_data/{test_dataset.__class__.__name__}"
        makedirs(save_dir, exist_ok=True)
        print(test_dataset.datapath)

        ids = np.random.choice(test_dataset.datadict['img_path'].shape[0],1000,replace=False)

        load_path = path.join(test_dataset.datapath, f'{test_dataset.metafilename}.p')

        with open(load_path, 'rb') as f:
            complete_datadict = pickle.load(f)

        all_nn_ids = np.arange(test_dataset.data['img_path'].shape[0])

        test_ids = test_dataset.test_indices
        train_ids = test_dataset.train_indices


        # nns for test set
        in_ids = np.arange(test_dataset.datadict['img_path'].shape[0])
        test_func = partial(get_nn, dataset= test_dataset)
        nn_ids_test = parallel_data_prefetch(test_func,in_ids,70)
        all_nn_ids[test_ids] = nn_ids_test

        # nns fir train set
        train_func = partial(get_nn, dataset=train_dataset)
        in_ids = np.arange(train_dataset.datadict['img_path'].shape[0])
        nn_ids_train = parallel_data_prefetch(train_func,in_ids,70)
        all_nn_ids[train_ids] = nn_ids_train

        test_nn_dict = {'nn_ids': nn_ids_test, 'dataset_ids': test_ids}
        train_nn_dict = {'nn_ids': nn_ids_train, 'dataset_ids': train_ids}


        # complete_datadict = deepcopy(train_dataset.data)
        complete_datadict.update({'nn_ids': all_nn_ids})


        with open(path.join(train_dataset.datapath,'meta_frange_kp_weights_nn.p'),'wb') as f:
            pickle.dump(complete_datadict,f,protocol=pickle.HIGHEST_PROTOCOL)

        with open(path.join(train_dataset.datapath,'nn_ids_train.p'),'wb') as f:
            pickle.dump(train_nn_dict,f,protocol=pickle.HIGHEST_PROTOCOL)

        with open(path.join(train_dataset.datapath, 'nn_ids_test.p'), 'wb') as f:
            pickle.dump(test_nn_dict, f, protocol=pickle.HIGHEST_PROTOCOL)





    else:


        if config['data']['dataset'] == "IperDataset":
            datakeys.extend(['keypoints_abs','keypoint_poke','nn'])


        test_dataset = dset(transforms, datakeys, config["data"], train=False)





        save_dir = f"test_data/{test_dataset.__class__.__name__}"
        makedirs(save_dir, exist_ok=True)
        print(test_dataset.datapath)

        if test_dataset.yield_videos:
            def init_fn(worker_id):
                return np.random.seed(np.random.get_state()[1][0] + worker_id)


            if config['data']['zero_poke']:
                sampler = FixedLengthSampler(test_dataset, config['data']['batch_size'], shuffle=True,
                                             drop_last=True, weighting=test_dataset.obj_weighting,
                                             zero_poke=config['data']['zero_poke'], zero_poke_amount=config['data']["zero_poke_amount"])
                loader = DataLoader(test_dataset, batch_sampler=sampler, num_workers=0, worker_init_fn=init_fn)
            else:
                if test_dataset.obj_weighting:
                    sampler = WeightedRandomSampler(weights=test_dataset.datadict["weights"], num_samples=test_dataset.datadict["img_path"].shape[0])
                    loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], num_workers=0, sampler=sampler)
                else:
                    loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], num_workers=0, shuffle=True)

            # sampler = FixedLengthSampler(test_dataset, config['data']['batch_size'], shuffle=True,
            #                              drop_last=True, weighting=test_dataset.obj_weighting,
            #                              zero_poke=config['data']['zero_poke'], zero_poke_amount=config['data']["zero_poke_amount"])
            # loader = DataLoader(test_dataset, batch_sampler=sampler, num_workers=config['data']['n_workers'], worker_init_fn=init_fn)



            #
            n_logged = config["n_logged"]
            #
            for i, batch in enumerate(tqdm(loader)):



                if i >100:
                    break

                sample_ids = batch['sample_ids']
                imgs = batch["images"][:n_logged]
                src_img = imgs[:,0]
                tgt_img = imgs[:,-1]
                flow = batch["flow"][:n_logged]
                flow_orig = batch['original_flow'][:n_logged]
                complete_poke = batch["poke"]
                if isinstance(complete_poke,list):
                    poke=complete_poke[0][:n_logged]
                    poke_coords = complete_poke[1][:n_logged]
                else:
                    poke_coords = None
                    poke = complete_poke

                postfix = "weighted" if test_dataset.obj_weighting else "unweighted"

                if 'keypoint_poke' in batch:
                    nn_row = np.concatenate(list(((batch['nn'][0][:n_logged,0] + 1.)*127.5).permute(0,2,3,1).numpy().astype(np.uint8)),axis=1)
                    start_img_row = np.concatenate(list(((imgs[:,0] + 1.)*127.5).permute(0,2,3,1).numpy().astype(np.uint8)),axis=1)

                    savename_nn = path.join(save_dir, f"NN-IMG-{i}-{postfix}.png")
                    nn_grid = np.concatenate([start_img_row,nn_row],axis=0)
                    nn_grid = cv2.cvtColor(nn_grid,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(savename_nn,nn_grid)


                    savename_kps_vis = path.join(save_dir,f'vis_kps_row-{i}.png')
                    vis_kps(batch['keypoints_abs'][:,0],imgs[:,0],savename_kps_vis)


                    kp_poke, kp_poke_coords, poke_ids = batch['keypoint_poke'][:n_logged]

                    out_vid_kp_poke = make_flow_video_with_samples(imgs[:,0],kp_poke,[],imgs,flow,n_logged=min(n_logged,config["data"]["batch_size"]),
                                                       wandb_mode=False,poke_normalized=test_dataset.normalize_flows, poke_coords=kp_poke_coords)

                    savename = path.join(save_dir, f"example-KP_POKE-{i}-{postfix}.mp4")

                    writer = cv2.VideoWriter(
                        savename,
                        cv2.VideoWriter_fourcc(*"MP4V"),
                        5,
                        (out_vid_kp_poke.shape[2], out_vid_kp_poke.shape[1]),
                    )

                    # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

                    for frame in out_vid_kp_poke:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame)

                    writer.release()



                #weights = batch["poke"][:n_logged][1] if test_dataset.flow_weights else None
            #
                postfix = "weighted" if test_dataset.obj_weighting else "unweighted"
                # if weights is not None:
                #     imgs = get_patches(imgs,weights,config["data"],test_dataset.weight_value_flow)
                #     postfix = postfix + "_patched"
                out_vid = make_flow_video_with_samples(imgs[:,0],poke,[],imgs,flow,n_logged=min(n_logged,config["data"]["batch_size"]),
                                                       wandb_mode=False,poke_normalized=test_dataset.normalize_flows, poke_coords=  poke_coords)


            #     warping_test = make_flow_grid(src_img,flow,tgt_warped,tgt_img,n_logged=min(n_logged,config["training"]["batch_size"]))
            #     warping_test = cv2.cvtColor(warping_test,cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(path.join(save_dir,f'warping_test-{i}.png'),warping_test)
            #
                savename = path.join(save_dir,f"example-{i}-{postfix}.mp4")

                writer = cv2.VideoWriter(
                    savename,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    5,
                    (out_vid.shape[2], out_vid.shape[1]),
                )

                # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

                for frame in out_vid:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                writer.release()

        else:

            if config["data"]["zero_poke"]:
                sampler = FixedLengthSampler(test_dataset,config["data"]["batch_size"],shuffle=True,drop_last=False,weighting=test_dataset.obj_weighting,zero_poke=True,zero_poke_amount=config["data"]["zero_poke_amount"])
                loader = DataLoader(test_dataset,batch_sampler=sampler,num_workers=config["data"]["n_workers"])
            else:
                loader = DataLoader(test_dataset, batch_size=config["data"]["batch_size"], num_workers=config["data"]["n_workers"],shuffle=True)



            for i, batch in enumerate(tqdm(loader)):

                if i >config["max_samples"]:
                    break

                src_img = batch["images"][:,0]
                tgt_img = batch["images"][:,-1]

                if only_images:
                    src_img = np.concatenate(list(((src_img.permute(0,2,3,1).numpy()+1.)*127.5).astype(np.uint8)),axis=1)
                    save_img = cv2.cvtColor(src_img,cv2.COLOR_RGB2BGR)
                    save_name = path.join(save_dir, f"example_imgs-{i}.png")
                    cv2.imwrite(save_name,save_img)
                else:
                    poke_coords = batch["poke_coords"]
                    poke = batch["poke"]
                    flow = batch["flow"]
                    flow_orig = batch["original_flow"]

                    flow_grid = make_flow_img_grid(src_img,tgt_img,None, poke_coords if config["poke_coords"] else poke,flow,flow_original=flow_orig)

                    flow_grid = cv2.cvtColor(flow_grid,cv2.COLOR_RGB2BGR)
                    save_name = path.join(save_dir,f"examples_grid-{postfix}-{i}.png")
                    cv2.imwrite(save_name,flow_grid)




