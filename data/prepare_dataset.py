import os
import cv2
import re
import argparse
import torch
import numpy as np
from os import path, makedirs
import pickle
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import yaml
import multiprocessing as mp
from multiprocessing import Process
from functools import partial
from dotmap import DotMap
from torchvision import transforms as tt
import configparser


from utils.general import parallel_data_prefetch
from data import get_dataset
from data.helper_functions import preprocess_image

h36m_aname2aid = {name: i for i, name in enumerate(["Directions","Discussion","Eating","Greeting","Phoning",
                                                    "Posing","Purchases","Sitting","SittingDown","Smoking",
                                                    "Photo","Waiting","Walking","WalkDog","WalkTogether"])}
h36m_aname2aid.update({"WalkingTogether": h36m_aname2aid["WalkTogether"]})
h36m_aname2aid.update({"WalkingDog": h36m_aname2aid["WalkDog"]})
h36m_aname2aid.update({"TakingPhoto": h36m_aname2aid["Photo"]})


def _do_parallel_data_prefetch(func, Q, data, idx):
    # create dummy dataset instance

    # run prefetching
    res = func(data)
    Q.put([idx, res])
    Q.put("Done")

def get_image(vidcap, frame_number,spatial_size=None):
    vidcap.set(1, frame_number)
    _, img = vidcap.read()
    if spatial_size is not None and spatial_size != img.shape[0]:
        img=cv2.resize(img,(spatial_size,spatial_size),interpolation=cv2.INTER_LINEAR)
    return img

def process_video(f_name, args):
    from utils.flownet_loader import FlownetPipeline
    from utils.general import get_gpu_id_with_lowest_memory, get_logger_old


    target_gpus = None if len(args.target_gpus) == 0 else args.target_gpus
    gpu_index = get_gpu_id_with_lowest_memory(target_gpus=target_gpus)
    torch.cuda.set_device(gpu_index)

    #f_name = vid_path.split(vid_path)[-1]

    logger = get_logger_old(f"{gpu_index}")

    extract_device = torch.device("cuda", gpu_index.index if isinstance(gpu_index,torch.device) else gpu_index)

    # load flownet
    pipeline = FlownetPipeline()
    flownet = pipeline.load_flownet(args, extract_device)

    # open video
    base_raw_dir = args.raw_dir.split("*")[0]

    if not isinstance(f_name,list):
        f_name = [f_name]

    logger.info(f"Iterating over {len(f_name)} files...")
    for fn in tqdm(f_name,):
        if fn.startswith('/'):
            fn = fn[1:]
        vid_path = path.join(base_raw_dir, fn)
        # vid_path = f"Code/input/train_data/movies/{fn}"
        vidcap = cv2.VideoCapture()
        vidcap.open(vid_path)
        counter = 0
        while not vidcap.isOpened():
            counter += 1
            time.sleep(1)
            if counter > 10:
                raise Exception("Could not open movie")

        # get some metadata
        number_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #upright = height > widt

        # create target path if not existent
        if args.data.dataset == 'Human36mDataset':
            vid_name = fn.split('/')[-1]
            if 'ALL' in vid_name:
                continue

            action = vid_name.split(' ')[0] if ' ' in vid_name else vid_name.split('.')[0]

            same_action_videos =list(filter(lambda y : y.startswith(action) and re.search(r'\d+$', y.split('.')[0]) is not None,
                                                                                   map(lambda x: x.split('/')[-1],f_name)))

            subject = fn.split('/')[-2]

            if re.search(r'\d+$', fn.split('.')[0]) is not None:
                subaction_id = int(fn[-1])
            else:
                max_id = max(map(lambda z: int(z.split(' ')[-1].split('.')[0]), same_action_videos))
                if max_id ==2:
                    subaction_id = 1
                else:
                    subaction_id = 2

            cam_id = vid_name.split('.')[1]
            base_path = path.join(args.processed_dir,subject,f'{action}-{subaction_id}',cam_id)

        else:
            base_path = path.join(args.processed_dir, fn.split(".")[0])   #.replace(str,str(args.spatial_size)))
        # base_path = f"Code/input/train_data/images/{f_name.split('.')[0]}/"
        makedirs(base_path, exist_ok=True)

        delta = args.flow_delta
        diff = args.flow_max


        # begin extraction
        for frame_number in range(0, number_frames, args.frames_discr):
            # break if not enough frames to properly extract sequence
            if frame_number >= number_frames - diff * args.frames_discr:
                break
            first_fidx, second_fidx = frame_number, frame_number + diff * args.frames_discr
            image_target_file = path.join(base_path, f"frame_{frame_number}.png")
            # image_target_file = f"{base_path}frame_{frame_number}.png"
            # FRAME
            if not path.exists(image_target_file):
                # write frame itself
                img = get_image(vidcap, frame_number)
                if img is None:
                    continue
                # if upright:
                #     img = cv2.transpose(img)
                try:
                    if args.spatial_size is None:
                        success = cv2.imwrite(image_target_file, img)
                    else:
                        img_res = cv2.resize(img,(args.spatial_size,args.spatial_size), interpolation=cv2.INTER_LINEAR)
                        success = cv2.imwrite(image_target_file,img_res)
                except cv2.error as e:
                    print(e)
                    continue
                except Exception as ex:
                    print(ex)
                    continue

                # if success:
                #     logger.info(f'wrote img with shape {img.shape} to "{image_target_file}".')
            # FLOW
            for d in range(0, diff*args.frames_discr, delta*args.frames_discr):
                if second_fidx - d < number_frames:
                    flow_target_file = path.join(
                        base_path, f"prediction_{first_fidx}_{second_fidx-d}.flow"
                    )
                    if not os.path.exists(flow_target_file + ".npy"):
                        # predict and write flow prediction
                        img, img2 = (
                            get_image(vidcap, first_fidx),
                            get_image(vidcap, second_fidx - d),
                        )
                        image_target_file2 = path.join(base_path, f"frame_{second_fidx - d}.png")
                        if not path.exists(image_target_file2):
                            try:
                                if args.spatial_size is None:
                                    success = cv2.imwrite(image_target_file2, img2)
                                else:
                                    img_res2 = cv2.resize(img2, (args.spatial_size, args.spatial_size), interpolation=cv2.INTER_LINEAR)
                                    success = cv2.imwrite(image_target_file2, img_res2)
                            except cv2.error as e:
                                print(e)
                                continue
                            except Exception as ex:
                                print(ex)
                                continue

                        sample = pipeline.preprocess_image(img, img2, "BGR",spatial_size=args.input_size).to(
                            extract_device
                        )
                        prediction = (
                            pipeline.predict(flownet, sample[None],spatial_size=args.spatial_size)
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        np.save(flow_target_file, prediction)

        logger.info(
            f'Finish processing video sequence "{fn}".')

    return "Finish"

def extract(args):


    # if args.process_vids:

    base_dir = args.raw_dir.split("*")[0]
    if not args.raw_dir.endswith('*'):
        args.raw_dir =path.join(args.raw_dir,'*')
    data_names = [p.split(base_dir)[-1] for p in glob(args.raw_dir) if p.endswith(args.video_format)]

    # data_names = [d for d in data_names if d in ['/VID_0_5.mkv','/VID_7_0.mkv']]



    fn_extract = partial(process_video, args=args)

    Q = mp.Queue(1000)
    # step = (
    #     int(len(data_names) / args.num_workers + 1)
    #     if len(data_names) % args.num_workers != 0
    #     else int(len(data_names) / args.num_workers)
    # )

    splits = np.array_split(np.arange(len(data_names)), args.num_workers)
    arguments = [
        [fn_extract, Q, part, i]
        for i, part in enumerate(
            [data_names[s[0]:s[-1]+1] for s in splits]
        )
    ]
    processes = []
    for i in range(args.num_workers):
        p = Process(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    start = time.time()
    gather_res = [[] for _ in range(args.num_workers)]
    try:
        for p in processes:
            p.start()
            time.sleep(20)

        k = 0
        while k < args.num_workers:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

def prepare(args):
    logger = get_logger_old("dataset_preparation")

    path2data = f'data/{args.data.dataset}'

    # set symbolic link to data directory
    if not path.islink(path2data):
        os.system(f'ln -s {args.processed_dir} {path2data}')


    datadict = {
        "img_path": [],
        "flow_paths": [],
        "fid": [],
        "vid": [],
        "img_size": [],
        "flow_size": [],
        "object_id":[],
    }
    if "iPER" in args.processed_dir.split("/") or "human36m" in args.processed_dir.split("/") or \
            "human3.6M" in args.processed_dir.split("/") :
        datadict.update({"action_id": [], "actor_id": []})

    train_test_split = args.data.dataset == 'Human36mDataset' or args.data.dataset == 'TaichiDataset'

    fmax = args.flow_max
    fdelta = args.flow_delta
    fd = args.frames_discr

    if train_test_split:
        datadict.update({"train": []})
        if args.data.dataset == 'TaichiDataset':
            oname2oid = {}

    # logger.info(f'Metafile is stored as "{args.meta_file_name}.p".')
    # logger.info(f"args.check_imgs is {args.check_imgs}")
    max_flow_length = int(fmax / fdelta)

    # if args.process_vids:
    if train_test_split:
        if args.data.dataset == 'Human36mDataset':
            videos = [d for d in glob(path.join(args.processed_dir, "*", "*", '*')) if path.isdir(d)]
        else:
            videos = [d for d in glob(path.join(args.processed_dir, "*", "*")) if path.isdir(d)]
    else:
        videos = [d for d in glob(path.join(args.processed_dir, "*")) if path.isdir(d)]

    videos = natsorted(videos)

    actual_oid = 0
    for vid, vid_name in enumerate(videos):
        logger.info(f'Video name is "{vid_name}"')
        images = glob(path.join(vid_name, "*.png"))
        images = natsorted(images)

        actor_id = action_id = train = None
        if args.data.dataset == 'PlantDataset':
            object_id = int(vid_name.split("/")[-1].split("_")[1])
        elif args.data.dataset == 'IperDataset':
            object_id = 100 * int(vid_name.split("/")[-1].split("_")[0]) + int(vid_name.split("/")[-1].split("_")[1])
            actor_id = int(vid_name.split("/")[-1].split("_")[0])
            action_id = int(vid_name.split("/")[-1].split("_")[-1])
        elif args.data.dataset == 'TaichiDataset':
            train = "train" == vid_name.split("/")[-2]
            msg = "train" if train else "test"
            print(f"Video in {msg}-split")

            obj_name = vid_name.split("/")[-1].split("#")[0]
            if obj_name in oname2oid.keys():
                object_id = oname2oid[obj_name]
            else:
                object_id = actual_oid
                oname2oid.update({obj_name: actual_oid})
                actual_oid += 1
        elif args.data.dataset == 'Human36mDataset':

            actor_id = int(vid_name.split('/')[-3][1:])
            object_id = actor_id
            action_name = vid_name.split('/')[-2].split('-')[0]
            action_id =  h36m_aname2aid[action_name]
            train = actor_id not in [9,11]
        else:
            raise ValueError("invalid dataset....")

        # max_flow_id = [len(images) - flow_step -1 for flow_step in range(fdelta*fd,fmax*fd+1, fdelta*fd)]

        for i, img_path in enumerate(
                tqdm(
                    images,
                    desc=f'Extracting meta information of video "{vid_name.split(args.processed_dir)[-1]}"',
                )
        ):
            fid = int(img_path.split("_")[-1].split(".")[0])
            #search_pattern = f'[{",".join([str(fid + n) for n in range(args.flow_delta,args.flow_max + 1, args.flow_delta)])}]'

            flows = natsorted([s for s in glob(path.join(vid_name, f"prediction_{fid}_*.npy"))
                               if (int(s.split("_")[-1].split(".")[0]) - int(s.split("_")[-2])) % (fdelta * fd) == 0 and
                               int(s.split("_")[-1].split(".")[0]) - int(s.split("_")[-2]) <= fmax*fd])

            # skip example if second image path does not exist
            if any(map(lambda p: not path.isfile(path.join(vid_name, f'frame_{p.split("_")[-1].split(".")[0]}.png')),flows)):
                logger.info(f'Breaking meta file information processing earlier for video "{vid_name.split("/")[-1]}", since not all image frames have been extracted.')
                break

            # make relative paths
            img_path_rel = img_path.split(args.processed_dir)[1]
            flows_rel = [f.split(args.processed_dir)[1] for f in flows]
            # filter flows
            flows_rel = [f for f in flows_rel if (int(f.split("/")[-1].split(".")[0].split("_")[-1]) - int(f.split("/")[-1].split(".")[0].split("_")[-2])) <= fmax*fd]

            if len(flows_rel) < max_flow_length:
                diff = max_flow_length-len(flows_rel)
                [flows_rel.insert(len(flows_rel),last_flow_paths[len(flows_rel)]) for _ in range(diff)]

            w_img = args.spatial_size
            h_img = args.spatial_size
            if len(flows) > 0:
                w_f = args.spatial_size
                h_f = args.spatial_size
            else:
                h_f = w_f = None

            assert len(flows_rel) == max_flow_length
            datadict["img_path"].append(path.join(path2data,img_path_rel[1:] if img_path_rel.startswith('/') else img_path_rel))
            fp_final = [path.join(path2data,fr[1:] if fr.startswith('/') else fr) for fr in flows_rel]
            datadict["flow_paths"].append(fp_final)
            datadict["fid"].append(fid)
            datadict["vid"].append(vid)
            # image size compliant with numpy and torch
            datadict["img_size"].append((h_img, w_img))
            datadict["flow_size"].append((h_f, w_f))
            datadict["object_id"].append(object_id)
            # datadict["max_fid"].append(max_flow_id)
            if action_id is not None:
                datadict["action_id"].append(action_id)
            if actor_id is not None:
                datadict["actor_id"].append(actor_id)
            if train is not None:
                datadict["train"].append(train)

            last_flow_paths = flows_rel

    logger.info(f'Prepared dataset consists of {len(datadict["img_path"])} samples.')

    # Store data (serialize)
    save_path = path.join(
        args.processed_dir, "meta.p"
    )


    with open(save_path, "wb") as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_nn(ids,train, cfg_dict):
    dset, transforms = get_dataset(cfg_dict)

    datakeys = ["images", "flow", "poke", "original_flow", 'sample_ids']

    dataset = dset(transforms, datakeys, cfg_dict, train=train)

    msg_train = 'train' if train else 'test'

    # compute nearest neighbours of each image frame based on the esimated keypoints

    msg = 'posture'
    def measure(idx):
        vid = dataset.datadict['vid'][idx]
        kps = dataset.datadict['keypoints_rel']
        sorted_ids = np.argsort(np.linalg.norm(kps[idx][None] - kps, axis=-1).sum(-1))
        # get only nearest neighbours from different video (to avoid nearest neighbour defined by preceding or sudceding frame in the same video)
        indicator_arr = dataset.datadict['vid'][sorted_ids] != vid
        nearest_with_other_video = np.flatnonzero(indicator_arr)[0]
        nn_sa = sorted_ids[nearest_with_other_video]
        nn_gen= sorted_ids[1]

        return nn_gen, nn_sa


    nn_other_vid = np.zeros((ids.shape[0]),dtype=int)
    nn_general = np.zeros((ids.shape[0]), dtype=int)

    print(f'Start NN computation for {nn_general.shape[0]} datapoints ')

    for c,i in enumerate(tqdm(ids,desc=f'Computing nearest neighbours in {msg} space for {msg_train}-set of {dataset.__class__.__name__}')):


        nn_gen, nn_ov = measure(i)

        nn_general[c] = nn_gen
        nn_other_vid[c] = nn_ov

    print(f'Finished nearest neighbor computation')

    return nn_other_vid



def pose_estimation(args,cfg_dict):
    # estimate keypoints
    os.system(f'python -m models.pose_estimator.tools.infer '
          f'--cfg models/pose_estimator/experiments/mpii/resnet/res152_256x256_d256x3_adam_lr1e-3.yaml '
          f'TEST.MODEL_FILE logs/pose_estimator/pose_resnet_152_256x256.pth '
          f'GPUS {args.target_gpus[0]} DATASET.DATASET {args.data.dataset}')

    # find nearest neighbors
    # generate dataset
    dset, transforms = get_dataset(cfg_dict["data"])

    datakeys = ["images", "flow", "poke", "original_flow", 'sample_ids']

    test_dataset = dset(transforms, datakeys, cfg_dict["data"], train=False)
    train_dataset = dset(transforms, datakeys, cfg_dict["data"], train=True)

    # ids = np.random.choice(test_dataset.datadict['img_path'].shape[0], 1000, replace=False)

    load_path = path.join(test_dataset.datapath, f'{test_dataset.metafilename}.p')

    with open(load_path, 'rb') as f:
        complete_datadict = pickle.load(f)

    all_nn_ids = np.arange(test_dataset.data['img_path'].shape[0])

    test_ids = test_dataset.test_indices
    train_ids = test_dataset.train_indices

    # nns for test set
    in_ids = np.arange(test_dataset.datadict['img_path'].shape[0])
    # test_func = partial(get_nn, dataset=test_dataset)
    data_test = (in_ids, False, cfg_dict['data'])
    nn_ids_test = parallel_data_prefetch(get_nn, data_test, 10)
    all_nn_ids[test_ids] = nn_ids_test

    # nns fir train set
    # train_func = partial(get_nn, dataset=train_dataset)
    in_ids = np.arange(train_dataset.datadict['img_path'].shape[0])
    data_train = (in_ids, True, cfg_dict['data'])
    nn_ids_train = parallel_data_prefetch(get_nn, data_train, 20)
    all_nn_ids[train_ids] = nn_ids_train

    test_nn_dict = {'nn_ids': nn_ids_test, 'dataset_ids': test_ids}
    train_nn_dict = {'nn_ids': nn_ids_train, 'dataset_ids': train_ids}

    # complete_datadict = deepcopy(train_dataset.data)
    complete_datadict.update({'nn_ids': all_nn_ids})

    with open(path.join(train_dataset.datapath, 'meta_kp_nn.p'), 'wb') as f:
        pickle.dump(complete_datadict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path.join(train_dataset.datapath, 'nn_ids_train.p'), 'wb') as f:
        pickle.dump(train_nn_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path.join(train_dataset.datapath, 'nn_ids_test.p'), 'wb') as f:
        pickle.dump(test_nn_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    import time
    from utils.general import get_logger_old



    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',type=str,required=True,help='Config file containing all parameters.')
    config_args = parser.parse_args()

    fpath = path.dirname(path.realpath(__file__))
    configfile = path.abspath(path.join(fpath,f'../{config_args.config}'))

    with open(configfile,'r') as f:
        args = yaml.load(f,Loader=yaml.FullLoader)
        cfg_dict = args

    args = DotMap(args)

    if args.data.dataset == 'Human36mDataset':
        h36config = configparser.ConfigParser()
        h36config.read(path.join(fpath, 'config.ini'))
        args.raw_dir = path.join(h36config['General']['TARGETDIR'], 'videos','*','*')
    #
    # cfg_dict['data']['datapath'] = args.processed_dir



    if args.raw_dir == '':
        raise ValueError(f'The data holding directory is currently not defined. please define the field "raw_dir" in  "{config_args.config}"')

    if args.processed_dir == '':
        raise ValueError(f'The target directory for the extracted image frames and flow maps is currently undefined. Please define the field "processed_dir" in  "{config_args.config}"')

    pool = []
    torch.multiprocessing.set_start_method("spawn")

    if args.mode == "extract":
        extract(args)
    elif args.mode == "prepare":  # in this case, it is prepare
        prepare(args)
    elif args.mode == 'pose_estimation':
        if args.data.dataset == 'IperDataset':
            pose_estimation(args,cfg_dict)
        else:
            raise NotImplementedError('Pose Estimation currently only supported for the IperDataset')
    elif args.mode == 'all':
        extract(args)
        prepare(args)
        if args.data.dataset == 'IperDataset':
            pose_estimation(args, cfg_dict)
    else:
        raise ValueError(f'The "mode"-parameter in config file "{configfile}" must be in [all, extract, prepare, pose_estimation], but is actually "{args.mode}"...')
