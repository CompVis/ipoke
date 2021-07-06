from experiments.experiment import Experiment#
from functools import partial

# from models.baselines.poke_encoder_fc import PokeAE
from models.conv_poke_encoder import ConvPokeAE
from data.datamodule import StaticDataModule


class PokeEncoderModel(Experiment):


    def __init__(self,config,dirs,devices):
        super().__init__(config,dirs,devices)

        # intiliaze models
        self.datakeys = ["poke","flow","images","original_flow"]



        self.config["architecture"].update({"in_size": self.config["data"]["spatial_size"][0]})

        model = ConvPokeAE

        if self.config["general"]["restart"]:
            ckpt_path = self._get_checkpoint()
            self.ae = model.load_from_checkpoint(ckpt_path,map_location="cpu",config=self.config)
        else:
            self.ae = model(self.config)
        # basic trainer is initialized in parent class
        # self.logger.info(
        #     f"Number of trainable parameters in model is {sum(p.numel() for p in self.ae.parameters())}"
        # )

        self.ckpt_callback = self.ckpt_callback(filename='{epoch}-{lpips-val:.3f}',monitor='lpips-val',
                                                save_top_k=self.config["logging"]["n_saved_ckpt"], mode='min')
        to_yaml_cb = self.add_ckpt_file()

        callbacks = [self.ckpt_callback,to_yaml_cb]
        if self.config["general"]["restart"] and ckpt_path is not None:
            self.basic_trainer = partial(self.basic_trainer,resume_from_checkpoint=ckpt_path,callbacks=callbacks)
        else:
            self.basic_trainer = partial(self.basic_trainer,callbacks=callbacks)




    def train(self):
        # prepare data
        datamod = StaticDataModule(self.config["data"],datakeys=self.datakeys)
        datamod.setup()
        n_batches_complete_train = len(datamod.train_dataloader())
        n_batches_complete_val = len(datamod.val_dataloader())
        n_train_batches = self.config["training"]["max_batches_per_epoch"] if n_batches_complete_train > self.config["training"]["max_batches_per_epoch"] else n_batches_complete_train
        n_val_batches = self.config["training"]["max_val_batches"] if n_batches_complete_val > self.config["training"]["max_val_batches"] else n_batches_complete_val

        if not self.is_debug:
            trainer = self.basic_trainer(limit_train_batches=n_train_batches, limit_val_batches=n_val_batches, limit_test_batches=n_val_batches)
        else:
            trainer = self.basic_trainer()

        trainer.fit(self.ae,datamodule=datamod)





    def test(self):
        pass
