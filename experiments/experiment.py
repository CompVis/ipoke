from abc import abstractmethod
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import PassThroughProfiler,AdvancedProfiler, SimpleProfiler
from pytorch_lightning.callbacks import ModelCheckpoint
from functools import partial
import yaml
import math
from glob import glob
from os import path
import numpy as np

from utils.general import get_logger
from utils.callbacks import BestCkptsToYaml

WANDB_DISABLE_CODE = True

class Experiment:
    def __init__(self, config:dict, dirs: dict, devices):
        super().__init__()
        self.config = config
        self.dirs = dirs
        self.devices = devices
        #self.logger = get_logger()

        ########## seed setting ##########

        seed = self.config['testing']['seed'] if 'seed' in self.config['testing'] and self.config['general']['test'] != 'none' else self.config['general']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)



        self.is_debug = self.config["general"]["debug"]
        if self.is_debug:
            self.config["data"]["n_workers"] = 0
            self.config["logging"]["n_samples_umap"] = 20
            self.config["logging"]["log_train_prog_at"] = 10
            self.config["data"]["batch_size"] = 2

        is_target_version = "target_version" in self.config["general"] and path.isdir(path.join(self.dirs["ckpt"],str(self.config["general"]["target_version"])))

        # wandb logging
        self.current_version = 0
        if path.isdir(path.join(self.dirs["ckpt"])):
            runs = glob(path.join(self.dirs["ckpt"],"*"))
            if len(runs) > 0:
                self.current_version = max([int(r.split("/")[-1]) for r in runs])
                if self.config["general"]["test"] == 'none':
                    self.current_version+=1


        if self.config["general"]["test"] != 'none' and is_target_version:
            self.current_version = self.config["general"]["target_version"]

        if self.config['general']['test'] == 'none':
            logger = WandbLogger(name=self.config["general"]["model_name"], save_dir=self.dirs["log"],
                              project="poking_inn",group=self.config["general"]["experiment"],tags=[self.config["data"]["dataset"]],
                             version=self.config["general"]["experiment"]+ "-" +self.config["general"]["model_name"]+ "-" + str(self.current_version),
                             save_code=True,entity='inn_poking')
        else:
            logger = False

        self.config["general"].update({"version":self.current_version})
        if self.config["general"]["restart"] or self.config["general"]["test"] != 'none':
            if is_target_version:
                self.ckpt_load_dir = path.join(self.dirs["ckpt"],str(self.config["general"]["target_version"]))
            else:
                if self.config["general"]["test"] != 'none':
                    self.ckpt_load_dir = path.join(self.dirs["ckpt"],str(self.current_version))
                else:
                    self.ckpt_load_dir = self.__get_latest_ckpt_dir()


        acc_batches = int(math.ceil(self.config["training"]["min_acc_batch_size"] / self.config["data"]["batch_size"])) \
            if self.config["training"]["min_acc_batch_size"] > self.config["data"]["batch_size"] else 1

        prof_file = path.join(self.dirs['log'],'profile.log')
        profiler =AdvancedProfiler(output_filename=prof_file) if self.config["general"]["profiler"] else None
        self.basic_trainer = partial(pl.Trainer, deterministic=True, gpus=devices, logger=logger,
                                    progress_bar_refresh_rate=1,profiler=profiler,
                                     accumulate_grad_batches=acc_batches,
                                     max_epochs=self.config["training"]["n_epochs"])
        if self.is_debug:
            self.basic_trainer = partial(self.basic_trainer,limit_train_batches=10,limit_val_batches=2,
                                         limit_test_batches=5,weights_summary="top",log_every_n_steps=2,num_sanity_val_steps=2)

        else:
            self.basic_trainer = partial(self.basic_trainer,val_check_interval=self.config["training"]["val_every"],num_sanity_val_steps=0)

        self.ckpt_callback = partial(ModelCheckpoint,dirpath=path.join(self.dirs["ckpt"],str(self.current_version)), period= 1,save_last=True)

        if 'test_batch_size' not in self.config['data']:
            self.config['data']['test_batch_size'] = 16

        if self.config['general']['test'] == 'none':
            logger.log_hyperparams(self.config)

        # signal.signal(signal.SIGINT,self.ckpt_to_yaml)

    def _get_checkpoint(self):


        ckpt_name = glob(path.join(self.ckpt_load_dir, "*.yaml"))
        last_ckpt = path.join(self.ckpt_load_dir, "last.ckpt")
        if self.config["general"]["last_ckpt"] and path.isfile(last_ckpt):
            print('Using last ckpt...')
            ckpt_name = [last_ckpt]
        elif self.config["general"]["last_ckpt"] and not path.isfile(last_ckpt):
            raise ValueError("intending to load last ckpt, but no last ckpt found. Aborting....")

        if len(ckpt_name) == 1:
            ckpt_name = ckpt_name[0]
        else:
            msg = "Not enough" if len(ckpt_name) < 1 else "Too many"
            raise ValueError(msg + f" checkpoint files found! Aborting...")

        if ckpt_name.endswith(".yaml"):
            with open(ckpt_name, "r") as f:
                ckpts = yaml.load(f, Loader=yaml.FullLoader)

            has_files = len(ckpts) > 0
            while has_files:
                best_val = min([ckpts[key] for key in ckpts])
                ckpt_name = {ckpts[key]:key for key in ckpts}[best_val]
                if path.isfile(ckpt_name):
                    break
                else:
                    del ckpts[ckpt_name]
                    has_files = len(ckpts) > 0

            if not has_files:
                raise ValueError(f'No valid files contained in ckpt-name-holding file "{ckpt_name}"')

        ckpt_file_name = ckpt_name.split('/')[-1]
        print(f'********************* Loading checkpoint for run_version #{self.current_version} with name "{ckpt_file_name}" *******************************')
        return ckpt_name

    def add_ckpt_file(self):
        assert isinstance(self.ckpt_callback,ModelCheckpoint)
        return BestCkptsToYaml(self.ckpt_callback)

    def __get_latest_ckpt_dir(self):
        start_version = self.current_version -1
        ckpt_dir = None
        for v in range(start_version,-1,-1):
            act_dir = path.join(self.dirs["ckpt"],str(v))
            if not path.isdir(act_dir):
                continue
            if self.config["general"]["last_ckpt"]:
                ckpt_dir = act_dir if path.isfile(path.join(act_dir,"last.ckpt")) else None
                print('Using last ckpt...')
            else:
                ckpt_dir = act_dir if (path.isfile(path.join(act_dir,"best_k_models.yaml")) and len(glob(path.join(act_dir,"*.ckpt"))) > 0) else None

            if ckpt_dir is not None:
                break

        if ckpt_dir is None:
            raise NotADirectoryError("NO valid checkpoint dir found but model shall be restarted! Aborting....")

        # self.logger.info(f'load checkpoint from file: "{ckpt_dir}"')

        return ckpt_dir

    @abstractmethod
    def train(self):
        """
        Here, the experiment shall be run
        :return:
        """
        pass

    @abstractmethod
    def test(self):
        """
        Here the prediction shall be run
        :param ckpt_path: The path where the checkpoint file to load can be found
        :return:
        """
        pass
