import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QMessageBox
import numpy as np
import cv2
import torch
import argparse
from glob import glob
from os import listdir,path
import time
import os
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from utils.general import LoggingParent
from data import get_dataset
from models.second_stage_video import PokeMotionModel
from utils.logging import make_poke_img,scale_imgs,make_temporal_border, save_video


class Form(QtWidgets.QDialog,LoggingParent):
    def __init__(self, config, dir_structure):
        QtWidgets.QDialog.__init__(self)
        LoggingParent.__init__(self)
        self.config = config
        self.dirs = dir_structure
        self.display_image_w, self.display_image_h = self.config["ui"]["display_size"], self.config["ui"]["display_size"]
        self.dataset, self.transforms = self.init_dataset()
        self.target_img_size =self.dataset.config["spatial_size"][0] if self.dataset.scale_poke_to_res else 256
        self.input_w, self.input_h = self.dataset.config["spatial_size"]
        self.scale_w, self.scale_h = self.input_w/self.display_image_w, self.input_h/self.display_image_h
        self.spacing = 20
        self.fps = self.config["ui"]["fps"]
        self.input_seq_length = self.config["ui"]["seq_length_to_generate"] if 'seq_length_to_generate' in self.config['ui'] else self.config['data']['max_frames']
        self.show_id = self.config["ui"]["show_id"]
        self.interactive = self.config["ui"]["interactive"] if "interactive" in self.config["ui"] else False
        #self.actual_seq_len = self.dataset.min_frames
        #self.mag2len = {self.dataset.seq}
        self.current_video = None
        self.actual_id = None
        self.actual_length = None
        self.current_poke = None
        self.current_poke_coords = None
        self.same_img_count = 0
        self.start_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.logger.info(f"Startup date and time: {self.start_time}")
        self.save_fps = self.config['ui']['save_fps']


        self.actual_torch_image, self.actual_image = self.load_next_image()
        self.old_torch_image = self.actual_torch_image
        self.init_images()
        self.net = None
        self.__get_net_model()

    def numpy_to_qImage(self, np_image):
        np_image2 = np_image.astype(np.uint8)
        qimage = QtGui.QImage(np_image2,
                              np_image2.shape[1],
                              np_image2.shape[0],
                              QtGui.QImage.Format_RGB888)
        return qimage

    def _load_ckpt(self):
        is_target_version = "target_version" in self.config["general"] and path.isdir(
            path.join(self.dirs["ckpt"], str(self.config["general"]["target_version"])))

        if not is_target_version:
            if path.isdir(path.join(self.dirs["ckpt"])):
                runs = [r for r in glob(path.join(self.dirs["ckpt"], '*')) if path.isdir(r)]
                print(f'Available runs are {runs}')
                if len(runs) == 0:
                    raise FileNotFoundError(f'No valid project file found. Check, if run name "{self.config["general"]["model_name"]}" is a valid run in experiment "{self.config["general"]["experiment"]}"...')
            else:
                raise FileNotFoundError(
                    f'No valid project file found. Check, if run name "{self.config["general"]["model_name"]}" is a valid run in experiment "{self.config["general"]["experiment"]}"...')

            current_version = max([int(r.split("/")[-1]) for r in runs])
        else:
            current_version = self.config['general']['target_version']

        ckpt_load_dir =  path.join(self.dirs["ckpt"],str(current_version))

        ckpt_name = glob(path.join(ckpt_load_dir, "*.yaml"))
        last_ckpt = path.join(ckpt_load_dir, "last.ckpt")
        if self.config["general"]["last_ckpt"] and path.isfile(last_ckpt):
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
                ckpt_name = {ckpts[key]: key for key in ckpts}[best_val]
                if 'DATAPATH' in os.environ:
                    ckpt_name = path.join(os.environ['DATAPATH'],ckpt_name[1:])
                if path.isfile(ckpt_name):
                    break
                else:
                    del ckpts[ckpt_name]
                    has_files = len(ckpts) > 0

            if not has_files:
                raise ValueError(f'No valid files contained in ckpt-name-holding file "{ckpt_name}"')

        return ckpt_name

    def forward(self,img,poke,length):

        if self.net.embed_poke_and_image:
            poke = torch.cat([poke,img],dim=1)

        # always eval
        self.net.eval()
        with torch.no_grad():
            poke_emb, *_ = self.net.poke_embedder.encoder(poke)
            # do not sample, as this mapping should be deterministic

            if self.net.use_cond:
                if self.net.conditioner.be_deterministic:
                    cond, *_ = self.net.conditioner.encoder(img)
                else:
                    _, cond, _ = self.net.conditioner.encoder(img)


            spatial=self.net.first_stage_config['architecture']['min_spatial_size']
            flow_input = torch.randn((1,self.config['architecture']['flow_in_channels'],spatial,spatial),device=self.config['gpu']).detach()

            if self.net.use_cond:
                cond = torch.cat([cond, poke_emb], dim=1)
            else:
                cond = poke_emb

            out_motion = self.net.flow(flow_input,cond,reverse=True)

            seq = self.net.decode_first_stage(out_motion,img[:,None],length=length)

            return seq


    def __get_net_model(self):
        ckpt_path = self._load_ckpt()
        self.net = PokeMotionModel.load_from_checkpoint(ckpt_path,map_location="cpu",config=self.config,strict=False, dirs=self.dirs)
        self.net.to(self.config['gpu'])
        self.logger.info(f'Net model loaded successfully')



    def load_next_image(self):
        if self.config["ui"]["target_id"] is None:
            actual_id = int(np.random.choice(np.arange(self.dataset.datadict["img_path"].shape[0]),1))
        else:
            if isinstance(self.config["ui"]["target_id"],int):
                actual_id = self.config["ui"]["target_id"]
            else:
                assert isinstance(self.config["ui"]["target_id"],list)
                actual_id = int(np.random.choice(self.config["ui"]["target_id"],1))
        self.actual_id = actual_id
        actual_img_path = self.dataset.datadict["img_path"][actual_id]
        actual_image = cv2.imread(actual_img_path)
        actual_image = cv2.resize(
            actual_image, self.dataset.config["spatial_size"], cv2.INTER_LINEAR
        )
        actual_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB)
        actual_torch_image = self.transforms(actual_image).unsqueeze(0).to(self.config["gpu"])
        actual_image = cv2.resize(actual_image, (self.display_image_w, self.display_image_h))
        return actual_torch_image, actual_image

    def update_gt_img(self):
        self.same_img_count = 0
        self.actual_torch_image, self.actual_image = self.load_next_image()
        self.old_torch_image = self.actual_torch_image
        self.gt.set_image(self.actual_image,id=self.actual_id if self.show_id else None)
        self.update_pd(self.actual_image,id=self.actual_id if self.show_id else None)

    def reset_gt_img(self):
        self.gt.set_image(self.actual_image, id=self.actual_id if self.show_id else None)
        self.actual_torch_image = self.old_torch_image
        self.current_poke = None
        self.current_poke_coords = None
        self.update_pd(self.actual_image,id=self.actual_id if self.show_id else None)


    def make_padded_video(self,src,vid,poke,poke_coords,vid_uint=False,src_uint=False):
        # if any(map(lambda x: x is None,[poke_coords,vid,poke,src])):
        #     return None,None
        n_padded = 4
        if not src_uint:
            src = scale_imgs(src).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        src1_with_poke1, _ = make_poke_img(src, poke, False, poke_coords)

        if not vid_uint:
            vid = scale_imgs(vid).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
        first_pad = np.stack([src1_with_poke1] * n_padded, axis=1)
        vid_pad = np.concatenate([first_pad, src[:,None],vid],axis=1)
        out_vid = make_temporal_border(vid_pad,poke,1,True,n_padded,concat=False)
        #only one samples should be obtained, as bs = 1
        out_vid = out_vid[0]

        # construtct static enrollment plot
        enrollment = np.concatenate(list(out_vid),axis=1)

        return out_vid, enrollment

    def generate_gt_poke_vid(self,basepath):
        if self.same_img_count > 0:
            return
        self.logger.info("Generating GT poke video....")



        # if self.dataset.var_sequence_length:
        idx = self.actual_id
        length  = self.actual_length
        # if self.dataset.var_sequence_length:
        #     ids = (idx, length - self.dataset.min_frames -1)
        # else:
        ids = (idx, self.dataset.max_frames)


        # get only source image
        imgs = self.dataset._get_imgs(ids,None).to(self.config["gpu"]).unsqueeze(0)
        img = imgs[:,0]
        # set mask, if required
        self.dataset.mask = {}
        self.dataset._get_mask(ids)

        if self.same_img_count == 0:
            n_padded = 4
            self.logger.info("save ground truth vid and enrollment....")
            src = scale_imgs(img).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pad = np.stack([src] * n_padded, axis=1)
            gt = scale_imgs(imgs).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
            pad_last = np.stack([gt[:,-1]]* n_padded,axis=1)
            gt_pad =np.concatenate([pad, pad, gt, pad_last],axis=1)[0]
            gt_enrollment = np.concatenate(list(gt_pad),axis=1)
            savename_gt = path.join(basepath, f'gt_vid.mp4')
            save_video(gt_pad,savename_gt,fps=self.save_fps)

            savename_gt_en = savename_gt[:-4] + 'enrollment.png'
            gt_enrollment = cv2.cvtColor(gt_enrollment,cv2.COLOR_RGB2BGR)
            cv2.imwrite(savename_gt_en,gt_enrollment)





        if self.config["ui"]["gt_poke"]:
            # sample defined number of pokes for which a video will be synthesized
            for i in tqdm(range(self.config["ui"]["n_gt_pokes"]),desc=f'Generating {self.config["ui"]["n_gt_pokes"]} gt pokes for id {idx}'):
                # if self.dataset.flow_weights:
                #     poke, _, poke_targets = self.dataset._get_poke(ids,yield_poke_target=True)
                # else:
                poke, poke_coords = self.dataset._get_poke(ids,yield_poke_target=True)
                poke = poke.to(self.config["gpu"]).unsqueeze(0)
                poke_coords = poke_coords[None]
                vid = self.forward(img,poke,length)
                # vid = ((vid + 1.) * 127.5).squeeze(0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


                act_video, enrollment = self.make_padded_video(img,vid,poke,poke_coords)
                savename = path.join(basepath, f'gt_poke_vid_{i}.mp4')
                save_video(act_video,savename,fps=self.save_fps)

                if self.config["ui"]["make_enrollment"]:
                    savename_en = savename[:-4] + "_enrollment.png"
                    e = cv2.cvtColor(enrollment, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(savename_en, e)
    


    def save_video(self):
        if self.current_video is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("No video was ever generated...")
            msg.setInformativeText("You cannot save an image before generating one!")
            msg.setWindowTitle("No video generated...")
            msg.exec_()
        else:
            if self.current_poke_coords is None or self.current_poke is None:
                self.logger.info('No poke and poke_coords defined. Nothing to save.')
                return

            vid = self.current_video[None]
            poke = self.current_poke
            poke_coords = self.current_poke_coords
            src = vid[:,0]

            self.logger.info(f"saving video with nr {self.same_img_count}...")
            basepath = path.join(self.dirs["generated"],"gui",f'id_{self.actual_id}',f'{self.start_time}')

            os.makedirs(basepath,exist_ok=True)
            savename = path.join(basepath,f'vid_{self.same_img_count}.mp4')

            out_vid, out_enrollment = self.make_padded_video(src,vid,poke,poke_coords,vid_uint=True,src_uint=True)

            save_video(out_vid,savename,fps=self.save_fps)

            if self.config["ui"]["make_enrollment"]:
                self.logger.info("Making enrollment plot...")
                savename_en = savename[:-4] + "_enrollment.png"
                e = cv2.cvtColor(out_enrollment, cv2.COLOR_RGB2BGR)
                cv2.imwrite(savename_en, e)

            self.generate_gt_poke_vid(basepath)

            self.same_img_count = self.same_img_count + 1





    def _generate_poke(self):
        source, target = self.gt.source, self.gt.target
        x_diff, y_diff = float(target.x() - source.x())/self.gt.max_amplitude, float(target.y() - source.y())/self.gt.max_amplitude
        # scale = np.sqrt(x_diff ** 2 + y_diff ** 2) / self.gt.max_amplitude * self.dataset.flow_cutoff
        # if self.dataset.var_sequence_length:
        #     x_poke, y_poke = x_diff * self.dataset.flow_cutoff, y_diff * self.dataset.flow_cutoff
        # else:
        x_poke = float(target.x() - source.x()) * ( self.target_img_size / self.display_image_w )
        y_poke = float(target.y() - source.y()) * ( self.target_img_size / self.display_image_h )

        poke_coords = torch.from_numpy(np.asarray([int(source.y() * self.scale_h),int(source.x() * self.scale_w)])[None,None])
        self.current_poke_coords = poke_coords

        length = self.input_seq_length
        poke = torch.zeros((2, self.input_h, self.input_w))
        half_poke_size = int(self.dataset.poke_size / 2)
        poke[0, int(source.y() * self.scale_h) - half_poke_size:int(source.y() * self.scale_h) + half_poke_size + 1,
                int(source.x() * self.scale_w) - half_poke_size:int(source.x() * self.scale_w) + half_poke_size + 1] = x_poke
        poke[1, int(source.y() * self.scale_h) - half_poke_size:int(source.y() * self.scale_h) + half_poke_size + 1,
                int(source.x() * self.scale_w) - half_poke_size:int(source.x() * self.scale_w) + half_poke_size + 1] = y_poke

        poke_final = poke.unsqueeze(0)
        self.current_poke = poke_final

        return poke_final.to(self.config["gpu"]), length

    def generate_sequence(self, path=""):
        print("Begin sequence generation")
        # get poke
        poke, self.actual_length = self._generate_poke()
        # x_diff = positive if source left and target right
        # y_diff = positive if source top and target bottom
        input_img = self.actual_torch_image.to(self.config["gpu"])

        seq = self.forward(input_img,poke,self.actual_length)

        self.actual_torch_image = seq[:,-1]

        seq = ((seq + 1.) * 127.5).squeeze(0).permute(0,2,3,1).cpu().numpy().astype(np.uint8)

        #seq = ((seq + 1.) * 127.5).squeeze(0).cpu().numpy().astype(np.uint8)

        #seq_debug = np.concatenate([np.stack([np.full_like(seq[0],255),np.zeros_like(seq[0])],0)] * 15,0)
        self.current_video = seq

        for i,img in enumerate(seq):
            self.gt.set_image(img, id=self.actual_id if self.show_id else None, sleep=True,draw=i<seq.shape[0]-1)
            # self.update_pd(img,id=self.actual_id if self.show_id else None)

        self.gt.set_image(seq[-1],id=self.actual_id if self.show_id else None, sleep=True)


    def update_pd(self,img, id=None):
        if img.shape[0] != self.display_image_h or img.shape[1] != self.display_image_w:
            img = cv2.resize(img,(self.display_image_h,self.display_image_w),interpolation=cv2.INTER_LINEAR)
        if self.show_id:
            if id is not None:
                img = cv2.UMat.get(cv2.putText(cv2.UMat(img), f"id {id}", (int(img.shape[1] // 3), img.shape[0] - int(img.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                               float(img.shape[0] / 256), (255, 0, 0), int(img.shape[0] / 128)))
        self.pd.setPixmap(
            QtGui.QPixmap(self.numpy_to_qImage(img.copy())).scaled(self.display_image_w, self.display_image_h)
        )
        time.sleep(1. / self.fps)
        QApplication.processEvents()

    def init_images(self):
        self.setWindowTitle("iPOKE UI")

        # Ground truth frame
        #scale with reference image size 256 as all datasets' flow estimates are of size 256
        max_amplitude=None

        self.gt = GTImage(self.display_image_w, self.display_image_h, self.actual_image, self,max_amplitude=max_amplitude,interactive_mode=self.interactive)
        self.gt.setGeometry(self.spacing, self.spacing, self.display_image_w, self.display_image_h)
        self.gt.set_image(self.actual_image, id=self.actual_id if self.show_id else None)
        self.gt_text = QtWidgets.QLabel(self)
        self.gt_text.setText("Generated Sequence")
        self.gt_text.setGeometry(self.spacing, 0, self.display_image_w, 20)

        # Predicted video
        self.pd = QtWidgets.QLabel(self)
        self.pd.setGeometry(self.spacing * 2 + self.display_image_w, self.spacing, self.display_image_w, self.display_image_h)
        self.pd.setPixmap(
            QtGui.QPixmap(self.numpy_to_qImage(self.actual_image)).scaled(self.display_image_w, self.display_image_h)
        )
        self.pd_text = QtWidgets.QLabel(self)
        self.pd_text.setText("Source Frame")
        self.pd_text.setGeometry(self.spacing * 2 + self.display_image_w, 0, self.display_image_w, 20)



        # finally gt and pd
        hbox = QtWidgets.QHBoxLayout()
        hbox2 = QtWidgets.QHBoxLayout()
        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)
        hbox.addWidget(self.gt_text)
        hbox.addWidget(self.pd_text)
        hbox2.addWidget(self.gt)
        hbox2.addWidget(self.pd)

        # Add a button to load next image in dataset
        btn2 = QtWidgets.QPushButton("Set to next Frame")
        btn2.clicked.connect(self.update_gt_img)
        btn2.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        generate_start = QtWidgets.QHBoxLayout()
        generate_start.addWidget(btn2)
        vbox.addLayout(generate_start)

        # Add a button to reset image in dataset
        btn3 = QtWidgets.QPushButton("Reset Frame")
        btn3.clicked.connect(self.reset_gt_img)
        btn3.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        generate_start = QtWidgets.QHBoxLayout()
        generate_start.addWidget(btn3)
        vbox.addLayout(generate_start)

        # add button to save generated video
        save_btn = QtWidgets.QPushButton("Save current Video")
        save_btn.clicked.connect(self.save_video)
        save_btn.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        generate_start = QtWidgets.QHBoxLayout()
        generate_start.addWidget(save_btn)
        vbox.addLayout(generate_start)

        # show all
        self.setLayout(vbox)
        self.show()

    def init_dataset(self):

        # important for sampling the right gt pokes afterwards
        self.config['data']['n_pokes']=1
        self.config['data']['filter'] = 'all'
        self.config['data']['fix_n_pokes'] = True

        dataset, transforms = get_dataset(config=self.config["data"])
        test_dataset = dataset(transforms, ["images","poke"], self.config["data"], train=False)
        if self.config["ui"]["target_id"] is not None and self.config["ui"]["write_path"]:
            if isinstance(self.config["ui"]["target_id"],int):
                ids = [self.config["ui"]["target_id"]]
            else:
                ids = self.config["ui"]["target_id"]

            self.logger.info("Write image paths....")
            savename = path.join(self.dirs["generated"],"image_files.txt")
            with open(savename,"w") as f:
                for idx in ids:
                    img_path = test_dataset.datadict["img_path"][idx]
                    f.write(img_path + "\n")


        return test_dataset, transforms

class GTImage(QtWidgets.QLabel):
    def __init__(self, display_image_w, display_image_h, g_img, parent, max_amplitude=None, interactive_mode = False):
        super().__init__()
        self.draw = False
        self.display_image_w, self.display_image_h = display_image_w, display_image_h
        self.ground_image = g_img
        self.source, self.target = None, None
        self.parent = parent
        self.parent.logger.info(f"Max ampltude of GTImage is {max_amplitude}")
        if max_amplitude==None:
            self.max_amplitude = int(display_image_w/5)
        else:
            self.max_amplitude = max_amplitude

        self.interactive = interactive_mode
        if self.interactive:
            self.parent.logger.info("Start GUI in interactive mode")

    def numpy_to_qImage(self, np_image):
        np_image = np_image.astype(np.uint8)
        qimage = QtGui.QImage(np_image.copy(),
                              np_image.shape[1],
                              np_image.shape[0],
                              QtGui.QImage.Format_RGB888)
        return qimage

    def set_image(self, img, id=None, sleep=False,draw=False):
        if img.shape[0] != self.display_image_h or img.shape[1] != self.display_image_w != self.display_image_w:
            img = cv2.resize(img, (self.display_image_h, self.display_image_w), interpolation=cv2.INTER_LINEAR)
        if id is not None:
            img = cv2.UMat.get(cv2.putText(cv2.UMat(img), f"id {id}", (int(img.shape[1] // 3), img.shape[0] - int(img.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                           float(img.shape[0] / 256), (255, 0, 0), int(img.shape[0] / 128)))

        if self.interactive and self.source is not None and self.target is not None and draw:
            img = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(img),
                                           (int(self.source.x()), int(self.source.y())),
                                           (int(self.target.x()), int(self.target.y())),
                                           (255, 0, 0),
                                           thickness=max(int(np.log2(self.display_image_w))-3,1)))
        self.ground_image = img
        self.setPixmap(
            QtGui.QPixmap(self.numpy_to_qImage(self.ground_image)).scaled(self.display_image_w, self.display_image_h)
        )
        if sleep:
            time.sleep(1. / self.parent.fps)
            QApplication.processEvents()


    def mousePressEvent(self, event):
        self.draw = True
        self.source = event.localPos()
        self.setCursor(QCursor(QtCore.Qt.PointingHandCursor))

    def mouseReleaseEvent(self, event):
        self.draw = False
        self.unsetCursor()
        self.parent.generate_sequence()

    def mouseMoveEvent(self, event):
        import copy
        pos = event.localPos()
        if pos.x() > 0.0 and pos.x() <= self.display_image_w and pos.y() > 0.0 and pos.y() <= self.display_image_h and self.draw:
            self.target = pos
            x_diff, y_diff = self.target.x() - self.source.x(), self.target.y() - self.source.y()
            amplitude = np.sqrt(x_diff**2 + y_diff**2)
            if amplitude > self.max_amplitude:
                scaler = self.max_amplitude/amplitude
                self.target.setX(int(self.source.x()+x_diff*scaler))
                self.target.setY(int(self.source.y()+y_diff*scaler))
            new_img = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(copy.deepcopy(self.ground_image)),
                                           (int(self.source.x()), int(self.source.y())),
                                           (int(self.target.x()), int(self.target.y())),
                                           (255, 0, 0),
                                           thickness=max(int(np.log2(self.display_image_w))-3,1)))
            self.setPixmap(
                QtGui.QPixmap(self.numpy_to_qImage(new_img)).scaled(self.display_image_w, self.display_image_h)
            )

def create_dir_structure(config, model_name):
    subdirs = ["ckpt", "config", "generated", "log"]
    structure = {subdir: path.join(config["general"]["base_dir"], config["general"]["experiment"], subdir, model_name) for subdir in subdirs}
    return structure


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config/second_stage.yaml",
                        help="Define config file")
    parser.add_argument("--gpu", default=[0], type=int,
                        nargs="+", help="GPU to use.")#parser.add_argument("--project_name","-n",type=str,required=True,help="The name of the project to be load.")
    parser.add_argument('-m',"--model_name", required=True, type=str, help="Name of the model that's intended to be used within the gui.")
    parser.add_argument("-si", "--show_id", default=False, action="store_true", help="Whether to display the actual id of the image or not.")
    parser.add_argument("-me","--make_enrollment",default=False, action="store_true", help="Make enrollment plot or not")
    parser.add_argument("-ds","--disc_step", default=1, type=int, help="discretization step for enrollments.")
    parser.add_argument("-gp", "--gt_poke", default=False, action="store_true", help="whether to output ground truth poke or not.")
    parser.add_argument("-pp", "--prepare_parts", default=False, action="store_true", help="whether to prepare parts or not.")
    parser.add_argument("-id", "--target_id", default=None, type=int,nargs="+", help="target od.")
    parser.add_argument("-wp", "--write_path", default=False, action="store_true", help="write image oaths or not.")
    parser.add_argument("-t", "--target_version", type=int, default=None, help="Target experiment version, if not specified, then last version is used.")
    parser.add_argument("-l", "--last_ckpt", default=False, action="store_true", help="Whether to use the last ckpt that was sotred during training.")
    #parser.add_argument("-np","--norm_percentile",type=int, default=50, choices=list(range(0,100,10)),help="The percentile of maxnorms of flow which shall be used for the input poke weighting for the model.")
    args = parser.parse_args()


    app = QtWidgets.QApplication(sys.argv)
    import yaml
    config_name = args.config
    with open(config_name,"r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)


    # this is for debug purposes on local machine
    if "BASEPATH_PREFIX" in os.environ:
        config["general"]["base_dir"] = os.environ["BASEPATH_PREFIX"] + config["general"]["base_dir"]
    elif "DATAPATH" in os.environ:
        config["general"]["base_dir"] = os.environ["DATAPATH"]+ config["general"]["base_dir"]#

    print(f'base dir is {config["general"]["base_dir"]}')


    #load actual model config for all fields bxut the "ui"-field
    dir_structure = create_dir_structure(config, args.model_name)
    saved_config = path.join(dir_structure["config"], "config.yaml")

    print(f'saved config is {saved_config}')

    if path.isfile(saved_config):
        with open(saved_config, "r") as f:
            complete_config = yaml.load(f, Loader=yaml.FullLoader)

    else:
        raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

    complete_config.update({"ui": config["ui"]})
    complete_config.update({"gpu": torch.device(
        f"cuda:{int(args.gpu[0])}"
        if torch.cuda.is_available() and int(args.gpu[0]) >= 0
        else "cpu"
    )})
    complete_config['testing'] = config['testing']

    if "DATAPATH" in os.environ:
        complete_config["general"]["base_dir"] = os.environ["DATAPATH"]+ complete_config["general"]["base_dir"]#

    if args.target_version is not None:
        complete_config['general'].update({'target_version': args.target_version})
    complete_config['general'].update({'last_ckpt': args.last_ckpt})
    complete_config['general'].update({'test': True})

    complete_config["ui"].update({"show_id": args.show_id})
    complete_config["ui"].update({"make_enrollment": args.make_enrollment})
    complete_config["ui"].update({"disc_step": args.disc_step})
    complete_config["ui"].update({"gt_poke": args.gt_poke})
    complete_config["ui"].update({"prepare_parts": args.prepare_parts})
    complete_config["ui"].update({"target_id":args.target_id})
    complete_config["ui"].update({"write_path": args.write_path})

    torch.cuda.set_device(complete_config["gpu"])
    if complete_config["ui"]["fixed_seed"]:
        ########## seed setting ##########
        torch.manual_seed(complete_config["general"]["seed"])
        torch.cuda.manual_seed(complete_config["general"]["seed"])
        np.random.seed(complete_config["general"]["seed"])
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(complete_config["general"]["seed"])
        rng = np.random.RandomState(complete_config["general"]["seed"])

    app_gui = Form(complete_config, dir_structure)
    app_gui.show()
    sys.exit(app.exec_())