import torch
import numpy as np
import argparse
from os import path
from tqdm import tqdm
import os

from utils.metrics import metric_vgg16, compute_div_score, compute_div_score_mse, compute_div_score_lpips
from utils.posenet_wrapper import PoseNetWrapper
from utils.general import get_logger_old



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", type=str,
                        default="/export/data/ablattma/visual_poking/savp/h36m_kth_params/samples_divscore/fake_samples.npy",
                        help="PAth to the samples file.")
    parser.add_argument("--gpu", type=int, required=True, help="The target device.")
    parser.add_argument('-r','--repr',type=str,default='vgg_features',choices=['keypoints','vgg_features'],help='The representation which shall be used for diversity calculation.')

    args = parser.parse_args()

    # name='' plans_test
    # path = '/export/data/ablattma/visual_poking/savp/h36m_kth_params/samples_divscore/fake_samples.npy'
        #'/export/scratch/mdorkenw/results/ICCV/diversity_srvp_iPER.npy'
        #'/export/data/ablattma/visual_poking/savp/plans_test /samples_divscore/fake_samples.npy'
    # device = 5
    if'DATAPATH' in os.environ:
        args.path = path.join(os.environ['DATAPATH'],args.path[1:])

    file = path.basename(__file__)
    logger = get_logger_old(file)


    videos = np.load(args.path)
    print(f'Range check before possible normalization! max: {videos.max()}; min: {videos.min()}')
    # videos shape is assumed to be (n_examples,n_samples_per_exmpl,sequence_length,channels, h,w)
    if videos.shape[0] < videos.shape[1]:
        videos = np.swapaxes(videos,0,1)

    if videos.max()>1.:
        videos = (videos.astype(float) / 127.5) - 1.

    if videos.shape[-1] == 3:
        videos = np.moveaxis(videos,(0,1,2,3,4,5),(0,1,2,4,5,3))

    assert videos.ndim == 6
    print(f'Range check after possible normalization! max: {videos.max()}; min: {videos.min()}')

    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    videos = torch.from_numpy(videos).to(torch.float32)
    if args.repr == 'vgg_features':



        logger.info('Using vgg features as similarity representation')
        vgg = metric_vgg16().to(dev)
        divl = compute_div_score(videos,vgg,device=dev)

        divl = np.asarray(divl).mean()
        logger.info(f'Average cosine distance in vgg features space {divl}')

    else:
        config = {'data':{'spatial_size': (videos.shape[-2],videos.shape[-1])}}
        posenet = PoseNetWrapper(config)
        posenet.eval()
        posenet.to(dev)
        logger.info('Using keypoints as similarity representation')

        n_ex, n_samples, seq_length, c, h, w = videos.shape

        divl = []
        with torch.no_grad():
            for video in tqdm(videos, f'Computing diversity score for {n_ex} examples with {n_samples} samples.'):

                video = video.to(dev).reshape(-1,*video.shape[2:])
                kps_raw = posenet(video)
                kps_abs, kps_rel = posenet.postprocess(kps_raw)

                for j in range(n_samples):
                    for k in range(n_samples):
                        if j != k:
                            f = kps_rel.reshape(n_samples, seq_length, *kps_rel.shape[1:])
                            divl.append(np.linalg.norm((f[j]-f[k]).reshape(-1,2)).mean())

        divl = np.asarray(divl).mean()
        logger.info(f'Average euclidean distance in keypoint space {divl}')

    div_score_mse = compute_div_score_mse(videos, device=dev)
    div_score_lpips = compute_div_score_lpips(videos, device=dev)

    text = f'Similarity measure_vgg: {divl}; similarity measure mse: {div_score_mse}; similarity measure lpips: {div_score_lpips}\n'

    print(text)

