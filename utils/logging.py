import torch
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import umap
import wandb
import os


def scale_imgs(x, input_format="float-1"):
    if input_format == "float-1":
        out = (x + 1.) * 127.5
    elif input_format == "float0":
        out = x * 255.
    else:
        raise ValueError(f'Specified Input Format "{input_format}" is invalid.')

    return out


def log_umap(z, z_m, z_p, logger: WandbLogger, step, title, ae_deterministic=False):
    umap_transform = umap.UMAP()
    transformation = umap_transform.fit(z_m)
    transformed_z = transformation.transform(z_m)

    plt.scatter(transformed_z[:, 0], transformed_z[:, 1], c='blue', s=1, marker='o', label="mean", alpha=.3, rasterized=True)
    plt.scatter(np.mean(transformed_z[:, 0]), np.mean(transformed_z[:, 1]), c='blue', s=20, marker='o', label="mean mean", alpha=.3)

    transformed_z = transformation.transform(z)
    plt.scatter(transformed_z[:, 0], transformed_z[:, 1], c='red', s=1, marker='v', label="INN samples", alpha=.3, rasterized=True)
    plt.scatter(np.mean(transformed_z[:, 0]), np.mean(transformed_z[:, 1]), c='red', s=20, marker='o', label="INN samples mean", alpha=.3)

    if not ae_deterministic:
        transformed_z = transformation.transform(z_p)
        plt.scatter(transformed_z[:, 0], transformed_z[:, 1], c='green', s=1, marker='s', label="posterior", alpha=.3, rasterized=True)
        plt.scatter(np.mean(transformed_z[:, 0]), np.mean(transformed_z[:, 1]), c='green', s=20, marker='o', label="posterior mean", alpha=.3)

    plt.legend()
    plt.axis('off')
    plt.ioff()
    logger.experiment.log({title: wandb.Image(plt, caption="Umap plot")}, step=step)
    plt.close()


def batches2image_grid(batches: list, captions=None, n_logged: int = None, image_range="float-1"):
    if n_logged is None:
        n_logged = batches[0].shape[0]

    if captions is None:
        captions = ["" for _ in range(len(batches))]

    row_list = []
    for imgs, caption in zip(batches, captions):

        imgs = scale_imgs(imgs[:n_logged].detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        img_row = np.concatenate([img for img in imgs], axis=1)
        if caption is not None:
            img_row = cv2.UMat.get(cv2.putText(cv2.UMat(img_row), caption, (int(img_row.shape[1] // 3), img_row.shape[0] - int(img_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                               float(img_row.shape[0] / 256), (255, 0, 0), int(img_row.shape[0] / 128)))

        row_list.append(img_row)

    return np.concatenate(row_list, axis=0)


def batches2flow_grid(flows: list, captions=None, n_logged: int = None, quiver=False, img=None, poke=None, poke_coords=None, poke_normalized=False):
    if n_logged is None:
        n_logged = flows[0].shape[0]

    vis_func = make_quiver_plot if quiver else vis_flow
    flow_list = []

    if img is not None:
        if poke is None:
            disp_img = np.concatenate(list(scale_imgs(img[:n_logged].detach()).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)), axis=1)
            img_cap = "Source image"
        else:
            img = scale_imgs(img[:n_logged].detach()).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            disp_img, _ = make_poke_img(img, poke[:n_logged], poke_coords=poke_coords if poke_coords is None else poke_coords[:n_logged]
                                        , poke_normalized=poke_normalized)
            disp_img = np.concatenate(disp_img, axis=1)
            img_cap = "Source image and poke"
        disp_img = cv2.UMat.get(cv2.putText(cv2.UMat(disp_img), img_cap, (int(disp_img.shape[1] // 3), disp_img.shape[0] - int(disp_img.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                            float(disp_img.shape[0] / 256), (255, 0, 0), int(disp_img.shape[0] / 128)))

        flow_list.append(disp_img)

    for flow, cap in zip(flows, captions):
        flow_vis = vis_func(flow[:n_logged].cpu().numpy())
        flow_row = np.concatenate(flow_vis, axis=1)
        if captions is not None:
            flow_row = cv2.UMat.get(cv2.putText(cv2.UMat(flow_row), cap, (int(flow_row.shape[1] // 3), flow_row.shape[0] - int(flow_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                float(flow_row.shape[0] / 256), (255, 0, 0), int(flow_row.shape[0] / 128)))

        flow_list.append(flow_row)

    return np.concatenate(flow_list, axis=0)


def put_text_to_video_row(video_row, text, color=None, display_frame_nr=False, n_padded=4):
    written = []
    for i, frame in enumerate(video_row):
        current = cv2.UMat.get(cv2.putText(cv2.UMat(frame), text, (int(frame.shape[1] // 3), frame.shape[0] - int(frame.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                           float(frame.shape[0] / 256), (255, 0, 0) if color is None else color, int(frame.shape[0] / 128)))
        if display_frame_nr:
            displayed_frame_nr = min(max(0, i - n_padded), video_row.shape[0] - 2 * n_padded)
            current = cv2.UMat.get(cv2.putText(cv2.UMat(current), str(displayed_frame_nr + 1), (int(frame.shape[1] / 32), frame.shape[0] - int(frame.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                               float(frame.shape[0] / 256), (255, 0, 0) if color is None else color, int(frame.shape[0] / 128)))

        written.append(current)
    return np.stack(written)


def make_poke_img(imgs, pokes, poke_normalized=False, poke_coords=None):
    pokes = pokes.cpu().numpy()
    raw_pokes = vis_flow(pokes)
    # is_poke_eq_img = len(pokes.shape) ==len(imgs.shape)
    if poke_coords is not None:
        poke_coords = poke_coords.detach().cpu().numpy()

    poke_imgs = []
    poke_vis = []
    for i, (poke, img) in enumerate(zip(pokes, imgs)):
        # if is_poke_eq_img:

        if poke_coords is not None:
            poke_vis_black = raw_pokes[i]
            for coord_pair in poke_coords[i]:
                if np.all(coord_pair > 0):
                    arrow_start = tuple(coord_pair)
                    arrow_dir = poke[:, arrow_start[0], arrow_start[1]]
                    if poke_normalized:
                        arrow_dir = arrow_dir / (np.linalg.norm(arrow_dir) + 1e-8) * (poke.shape[1] / 5)
                    if not math.isnan(arrow_dir[0]) or not math.isnan(arrow_dir[1]):
                        # reverse as opecv requires x coordinate first
                        arrow_start = tuple(reversed(arrow_start))
                        arrow_end = (arrow_start[0] + int(math.ceil(arrow_dir[0])), arrow_start[1] + int(math.ceil(arrow_dir[1])))
                        img = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(img), arrow_start, arrow_end, (255, 0, 0), max(int(img.shape[0] / 64), 1)))
                        poke_vis_black = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(poke_vis_black), arrow_start, arrow_end, (255, 0, 0), max(int(img.shape[0] / 64), 1)))
                    else:
                        continue

            poke_imgs.append(img)
            poke_vis.append(poke_vis_black)
        else:
            active = np.nonzero(pokes[i].any(0) > 0)
            if active[0].size == 0:
                # case: zero_poke
                poke_imgs.append(img)
                poke_vis.append(raw_pokes[i])
                continue
            min_y = np.amin(active[0])
            max_y = np.maximum(np.amax(active[0]), min_y + 1)
            min_x = np.amin(active[1])
            max_x = np.maximum(np.amax(active[1]), min_x + 1)
            active_poke = poke[:, min_y:max_y, min_x:max_x]
            if len(active_poke.squeeze().shape) == 1:
                avg_flow = active_poke
            else:
                avg_flow = np.mean(active_poke, axis=(1, 2))
            if poke_normalized:
                arrow_dir = avg_flow / (np.linalg.norm(avg_flow) + 1e-8) * (poke.shape[1] / 5)
            else:
                arrow_dir = avg_flow

            if not math.isnan(arrow_dir[0]) or not math.isnan(arrow_dir[1]):
                arrow_start = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
                arrow_end = (arrow_start[0] + int(math.ceil(arrow_dir[0])), arrow_start[1] + int(math.ceil(arrow_dir[1])))
            else:
                poke_imgs.append(img)
                continue

            poke_and_image = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(img), arrow_start, arrow_end, (255, 0, 0), max(int(img.shape[0] / 64), 1)))
            black_and_poke = cv2.UMat.get(cv2.arrowedLine(cv2.UMat(raw_pokes[i]), arrow_start, arrow_end, (255, 0, 0), max(int(img.shape[0] / 64), 1)))
            # else:
            #     # poke is (l_x,l_y,p_x,p_y)
            #     arrow_start = (int(poke[0]), int(poke[1]))
            #     if poke_normalized:
            #         arrow_dir = np.ceil(poke[2:4] / (np.linalg.norm(poke[2:4])+1e-8) * (img.shape[1] / 5))
            #     else:
            #         arrow_dir = (math.ceil(poke[2]),math.ceil(poke[3]))
            #
            #     arrow_end = (arrow_start[0]+int(arrow_dir[0]),arrow_start[1]+int(arrow_dir[1]))

            poke_imgs.append(poke_and_image)
            poke_vis.append(black_and_poke)

    return poke_imgs, poke_vis


def vis_flow(flow_map, normalize=False):
    if isinstance(flow_map, torch.Tensor):
        flow_map = flow_map.cpu().numpy()
    flows_vis = []
    for flow in flow_map:
        hsv = np.zeros((*flow.shape[1:], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[0], flow[1])
        # since 360 is not valid for uint8, 180° corresponds to 360° for opencv hsv representation. Therefore, we're dividing the angle by 2 after conversion to degrees
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        as_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if normalize:
            as_rgb = as_rgb.astype(np.float) - as_rgb.min(axis=(0, 1), keepdims=True)
            as_rgb = (as_rgb / as_rgb.max(axis=(0, 1), keepdims=True) * 255.).astype(np.uint8)
        flows_vis.append(as_rgb)

    return flows_vis


def make_quiver_plot(flow, step=4):
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    quivers = []
    for f in flow:
        fig, ax = plt.subplots(figsize=(float(f.shape[2]) / 100, float(f.shape[1]) / 100))
        ax.quiver(np.arange(0, f.shape[2], step), np.arange(f.shape[1] - 1, -1, -step),
                  f[0, ::step, ::step], f[1, ::step, ::step])  # ,cm.get_cmap("plasma",int(f.shape[1]/step))

        plt.axis("off")

        quiv = fig2data(fig, f.shape[1:])
        quivers.append(quiv)
        plt.close()

    return quivers


def make_flow_img_grid(start_img, tgt, samples, poke, flow, image_range="float-1", poke_normalized=False, flow_original=None):
    is_samples = samples is not None

    start_img = scale_imgs(start_img.detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    tgt = scale_imgs(tgt.detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    if is_samples:
        sample_rows = []
        for i, sample in enumerate(samples):
            sample = scale_imgs(sample.detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            sample_row = np.concatenate(list(sample), axis=1)
            sample_row = cv2.UMat.get(cv2.putText(cv2.UMat(sample_row), "Poke samples", (int(sample_row.shape[1] // 3), sample_row.shape[0] - int(sample_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(sample_row.shape[0] / 256), (255, 0, 0), int(sample_row.shape[0] / 128)))
            sample_rows.append(sample_row)

        sample_grid = np.concatenate(sample_rows, axis=0)

    poke_imgs, _ = make_poke_img(start_img, poke, poke_normalized)
    poke_img_row = np.concatenate(poke_imgs, axis=1)
    poke_img_row = cv2.UMat.get(
        cv2.putText(cv2.UMat(poke_img_row), "Pokes and Sources", (int(poke_img_row.shape[1] // 3), poke_img_row.shape[0] - int(poke_img_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    float(poke_img_row.shape[0] / 256), (255, 255, 255), int(poke_img_row.shape[0] / 128)))

    tgt_row = np.concatenate(list(tgt), axis=1)
    tgt_row = cv2.UMat.get(cv2.putText(cv2.UMat(tgt_row), "Targets", (int(tgt_row.shape[1] // 3), tgt_row.shape[0] - int(tgt_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                       float(tgt_row.shape[0] / 256), (255, 255, 255), int(tgt_row.shape[0] / 128)))

    flow_vis = vis_flow(flow)
    flow_vis = np.concatenate(list(flow_vis), axis=1)
    flow_vis = cv2.UMat.get(cv2.putText(cv2.UMat(flow_vis), "Optical Flow", (int(flow_vis.shape[1] // 3), flow_vis.shape[0] - int(flow_vis.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                        float(flow_vis.shape[0] / 256), (255, 255, 255), int(flow_vis.shape[0] / 128)))

    # generate quiver plot
    quiver = make_quiver_plot(flow)
    quiver = np.concatenate(quiver, axis=1)
    quiver = cv2.UMat.get(cv2.putText(cv2.UMat(quiver), "Quiver Plot", (int(quiver.shape[1] // 3), quiver.shape[0] - int(quiver.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                      float(quiver.shape[0] / 256), (255, 255, 255), int(quiver.shape[0] / 128)))

    if flow_original is not None:
        flow_orig = vis_flow(flow_original)
        flow_orig = np.concatenate(list(flow_orig), axis=1)
        flow_orig = cv2.UMat.get(cv2.putText(cv2.UMat(flow_orig), "Optical Flow Original", (int(flow_orig.shape[1] // 3), flow_orig.shape[0] - int(flow_orig.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                             float(flow_orig.shape[0] / 256), (255, 255, 255), int(flow_orig.shape[0] / 128)))

    if is_samples:
        out_list = [poke_img_row, tgt_row, sample_grid, flow_vis, quiver]
    else:
        out_list = [poke_img_row, tgt_row, flow_vis, quiver]

    if flow_original is not None:
        out_list.append(flow_orig)

    out_grid = np.concatenate(out_list, axis=0)

    return out_grid


def make_animated_grid(start_img, tgt, samples, poke, flow, image_range="float-1", poke_normalized=False, wandb_mode=True):
    flow_vis = vis_flow(flow)
    flow_vis = np.concatenate(list(flow_vis), axis=1)
    flow_vis = cv2.UMat.get(cv2.putText(cv2.UMat(flow_vis), "Optical Flow", (int(flow_vis.shape[1] // 3), flow_vis.shape[0] - int(flow_vis.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                        float(flow_vis.shape[0] / 256), (255, 255, 255), int(flow_vis.shape[0] / 128)))
    static_flow = np.stack([flow_vis] * 3, axis=0)

    quiver = make_quiver_plot(flow)
    quiver = np.concatenate(quiver, axis=1)
    quiver = cv2.UMat.get(cv2.putText(cv2.UMat(quiver), "Quiver Plot", (int(quiver.shape[1] // 3), quiver.shape[0] - int(quiver.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                      float(quiver.shape[0] / 256), (255, 255, 255), int(quiver.shape[0] / 128)))
    static_quiver = np.stack([quiver] * 3, axis=0)

    start_img = scale_imgs(start_img.detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    tgt = scale_imgs(tgt.detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    tgt_row = np.concatenate(list(tgt), axis=1)
    tgt_row = cv2.UMat.get(cv2.putText(cv2.UMat(tgt_row), "Targets", (int(tgt_row.shape[1] // 3), tgt_row.shape[0] - int(tgt_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                       float(tgt_row.shape[0] / 256), (255, 255, 255), int(tgt_row.shape[0] / 128)))

    poke_imgs = make_poke_img(start_img, poke, poke_normalized)
    poke_img_row = np.concatenate(poke_imgs, axis=1)
    poke_img_row = cv2.UMat.get(
        cv2.putText(cv2.UMat(poke_img_row), "Pokes and Sources", (int(poke_img_row.shape[1] // 3), poke_img_row.shape[0] - int(poke_img_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    float(poke_img_row.shape[0] / 256), (255, 255, 255), int(poke_img_row.shape[0] / 128)))

    animations = []
    for sample in samples:
        sample = scale_imgs(sample.detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        sample_row = np.concatenate(list(sample), axis=1)
        sample_row = cv2.UMat.get(cv2.putText(cv2.UMat(sample_row), "Poke samples", (int(sample_row.shape[1] // 3), sample_row.shape[0] - int(sample_row.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                              float(sample_row.shape[0] / 256), (255, 0, 0), int(sample_row.shape[0] / 128)))

        anim = np.stack([poke_img_row, sample_row, tgt_row], axis=0)
        animations.append(anim)

    animations = np.concatenate(animations + [static_flow, static_quiver], axis=1)
    return np.moveaxis(animations, [0, 1, 2, 3], [0, 2, 3, 1]) if wandb_mode else animations


def make_video(targets, preds, n_logged, log_wandb=True, n_max_per_row=6, show_frames=True):
    src = ((preds.permute(0, 1, 3, 4, 2).detach().cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    tgt = ((targets.permute(0, 1, 3, 4, 2).detach().cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]

    if n_logged > n_max_per_row:
        assert n_logged % n_max_per_row == 0
        src_rows = [np.concatenate(list(src[i * n_max_per_row:(i + 1) * n_max_per_row]), axis=2) for i in range(int(n_logged / n_max_per_row))]
        tgt_rows = [np.concatenate(list(tgt[i * n_max_per_row:(i + 1) * n_max_per_row]), axis=2) for i in range(int(n_logged / n_max_per_row))]
    else:
        src_rows = [np.concatenate(list(src), axis=2)]
        tgt_rows = [np.concatenate(list(tgt), axis=2)]

    sub_grids = []
    for src_row, tgt_row in zip(src_rows, tgt_rows):
        src_row = put_text_to_video_row(src_row, "Predicted Videos", display_frame_nr=show_frames)
        tgt_row = put_text_to_video_row(tgt_row, "Target videos", display_frame_nr=show_frames)
        sub_grids.append(np.concatenate([src_row, tgt_row], axis=1))

    full_grid = np.concatenate(sub_grids, axis=1) if len(sub_grids) > 1 else sub_grids[0]

    if log_wandb:
        full_grid = np.moveaxis(full_grid, [0, 1, 2, 3], [0, 2, 3, 1])

    return full_grid


def draw_poke_rect(imgs, pokes):
    if isinstance(pokes, torch.Tensor):
        pokes = pokes.detach().cpu().numpy()

    imgs_out = []

    for i, (img, poke) in enumerate(zip(imgs, pokes)):
        poke_points = np.nonzero((poke > 0).any(-1))
        if poke_points[0].size == 0:
            imgs_out.append(np.zeros_like(img))
        else:
            min_y = np.amin(poke_points[0])
            max_y = np.amax(poke_points[0])
            min_x = np.amin(poke_points[1])
            max_x = np.amax(poke_points[1])
            # draw rect
            img_with_rect = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 255, 255),
                                          max(1, int(img.shape[0] // 64)))
            imgs_out.append(img_with_rect)

    return imgs_out


def get_endpoint(poke, n_logged, poke_coords=None):
    if not isinstance(poke, np.ndarray):
        poke = poke.detach().cpu().numpy()

    endpoints = []
    for i, p in enumerate(poke[:n_logged]):
        current_endpoints = []
        if poke_coords is not None:

            for coord_pair in poke_coords[i]:
                if np.all(coord_pair > 0):
                    arrow_start = tuple(coord_pair)
                    arrow_dir = p[:, arrow_start[0], arrow_start[1]]
                    if not math.isnan(arrow_dir[0]) or not math.isnan(arrow_dir[1]):
                        # reverse as opecv requires x coordinate first
                        arrow_start = tuple(reversed(arrow_start))
                        endpoint = (arrow_start[0] + int(math.ceil(arrow_dir[0])), arrow_start[1] + int(math.ceil(arrow_dir[1])))
                        current_endpoints.append(endpoint)
                    else:
                        continue
        else:
            poke_points = np.nonzero(np.linalg.norm(p, axis=0) > 0)
            start_y = poke_points[0].mean()
            start_x = poke_points[1].mean()

            # idx_y = np.abs(poke_points[0]-start_y).argmin()
            # idx_x = np.abs(poke_points[1]-start_x).argmin()

            dir_x = p[0, int(start_y), int(start_x)]
            dir_y = p[1, int(start_y), int(start_x)]

            end_y = int(np.round(start_y + dir_y))
            end_x = int(np.round(start_x + dir_x))
            current_endpoints.append((end_x, end_y))

        endpoints.append(current_endpoints)

    return endpoints


def draw_endpoints_to_video(poke, videos, n_logged, poke_coords=None):
    endpoints = get_endpoint(poke, n_logged, poke_coords)

    endframes_with_poke = []
    for ep, vid in zip(endpoints, videos[:n_logged]):
        drawn_endframe = vid[-1]
        for p in ep:
            drawn_endframe = cv2.UMat.get(cv2.circle(cv2.UMat(drawn_endframe), p,
                                                     max(int(vid.shape[0] / 32), 2),
                                                     (255, 0, 0), thickness=-1))
        endframes_with_poke.append(drawn_endframe)

    return endframes_with_poke


def make_temporal_border(video, poke, n_logged, draw_endpoint=False, n_pad_frames=4, poke_coords=None,
                         startframe=None, concat=True):
    if poke_coords is not None:
        poke_coords = poke_coords.detach().cpu().numpy()

    if draw_endpoint:
        vid_endframes = draw_endpoints_to_video(poke, video, n_logged, poke_coords)
    else:
        vid_endframes = [vid[-1] for vid in video]

    if startframe is None:
        vid_startframes = [vid[0] for vid in video]
    else:
        if len(startframe.shape) == 4:
            startframe = startframe[0]
        vid_startframes = [startframe] * len(video)

    padded_vids = []
    for sf, ef, vid in zip(vid_startframes, vid_endframes, video):
        end_frames = np.stack([ef] * n_pad_frames, axis=0) if n_pad_frames > 1 else ef[None]
        start_frames = np.stack([sf] * n_pad_frames, axis=0) if n_pad_frames > 1 else sf[None]

        padded_vids.append(np.concatenate([start_frames, vid, end_frames], axis=0))

    # make video row
    if concat:
        return np.concatenate(padded_vids, axis=2)
    else:
        return padded_vids


def make_flow_video_with_samples(src, poke, samples, tgt, flow, n_logged, poke_normalized=False, wandb_mode=True, image_range="float-1", poke_coords=None):
    # prepare samples
    n_padded = 4
    samples = [torch.cat([src[:, None], s], dim=1).detach() for s in samples]
    samples = [scale_imgs(s, input_format=image_range).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)[:n_logged] for s in samples]

    start_img = scale_imgs(src.detach(), input_format=image_range).permute(0, 2, 3, 1).cpu().numpy().astype(
        np.uint8)[:n_logged]
    src_with_arr, poke_with_arr = make_poke_img(start_img, poke[:n_logged], poke_normalized, poke_coords)
    src_with_arr = [np.stack([s] * tgt.size(1), axis=0) for s in src_with_arr]
    poke_with_arr = [np.stack([p] * tgt.size(1), axis=0) for p in poke_with_arr]

    # don't draw poke for these
    src_arr_vid_row = make_temporal_border(src_with_arr, poke, n_logged, n_pad_frames=n_padded)
    poke_arr_vid_row = make_temporal_border(poke_with_arr, poke, n_logged, n_pad_frames=n_padded)
    # src_arr_vid_row = np.stack([np.concatenate(src_with_arr,axis=1)]*tgt.size(1),axis=0)
    src_arr_vid_row = put_text_to_video_row(src_arr_vid_row, "Input Image With Poke", display_frame_nr=False)
    # poke_arr_vid_row= np.stack([np.concatenate(poke_with_arr,axis=1)]*tgt.size(1),axis=0)
    poke_arr_vid_row = put_text_to_video_row(poke_arr_vid_row, "Poke", color=(255, 255, 255), display_frame_nr=False)

    # visualize flow in color plot
    flow_vis = vis_flow(flow[:n_logged])
    # flow_vis = draw_poke_rect(flow_vis,poke[:n_logged])
    flow_vis = [np.stack([f] * tgt.size(1), axis=0) for f in flow_vis]
    flow_vis_row = make_temporal_border(flow_vis, poke, n_logged, n_pad_frames=n_padded)
    flow_vis_row = put_text_to_video_row(flow_vis_row, 'Optical Flow', color=(255, 255, 255), display_frame_nr=False)

    # visualize flow in quiver plot
    quiver = make_quiver_plot(flow[:n_logged])
    quiver = [np.stack([q] * tgt.size(1), axis=0) for q in quiver]
    quiver_row = make_temporal_border(quiver, poke, n_logged, n_pad_frames=n_padded)
    quiver_row = put_text_to_video_row(quiver_row, 'Quiver Plot', display_frame_nr=False)

    tgt = scale_imgs(tgt.detach(), input_format=image_range).permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)[
          :n_logged]
    tgt_row = make_temporal_border(tgt, poke, n_logged, draw_endpoint=not poke_normalized, poke_coords=poke_coords)
    # tgt_row = np.concatenate(list(tgt),axis=2)
    tgt_row = put_text_to_video_row(tgt_row, 'Groundtruth Videos', display_frame_nr=True, n_padded=n_padded)

    sample_rows = []
    for i, sample in enumerate(samples):
        sample_row = make_temporal_border(sample, poke, n_logged, draw_endpoint=not poke_normalized,
                                          n_pad_frames=n_padded, poke_coords=poke_coords)
        sample_row = put_text_to_video_row(sample_row, f'Sample #{i + 1}', display_frame_nr=True, n_padded=n_padded)
        sample_rows.append(sample_row)

    full_grid = np.concatenate([src_arr_vid_row] + sample_rows + [tgt_row, flow_vis_row, poke_arr_vid_row, quiver_row], axis=1)

    if wandb_mode:
        full_grid = np.moveaxis(full_grid, [0, 1, 2, 3], [0, 2, 3, 1])

    return full_grid


def vis_kps(keypoints, imgs, savepath):
    imgs_kps = []

    for kps, img in zip(keypoints, imgs):

        img = ((img + 1.) * 127.5).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

        for idx, kp in enumerate(kps):
            img = cv2.UMat.get(cv2.circle(cv2.UMat(img), (int(kp[0]), int(kp[1])), 2, (255, 0, 255), -1, ))
            img = cv2.UMat.get(cv2.putText(cv2.UMat(img), f'{idx}', (int(kp[0]) + 5, int(kp[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                           float(img.shape[0] / 512), (255, 0, 0), int(img.shape[0] / 256)))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        imgs_kps.append(img)

    # make row
    imgs_kps = np.concatenate(imgs_kps, axis=1)

    cv2.imwrite(savepath, imgs_kps)


def get_divisors(N):
    candidate = np.ceil(np.sqrt(N))

    while N % candidate != 0:
        candidate += 1

    M = N // candidate

    return int(candidate), int(M)


def put_text(img, text, loc=None, color=None, font_scale=None):
    if loc is None:
        loc = (int(img.shape[1] // 3), img.shape[0] - int(img.shape[0] / 6))
    return cv2.UMat.get(cv2.putText(cv2.UMat(img), text, loc, cv2.FONT_HERSHEY_SIMPLEX,
                                    float(img.shape[0] / 256) if font_scale is None else font_scale, (255, 0, 0) if color is None else color, int(img.shape[0] / 128)))


def make_transfer_grids(src1, src2, poke1, poke2, vid1, vid2, m1_c2, m2_c1, poke_coords1, poke_coords2, poke_normalized=False, make_enrollment=False, sample_ids1=None, sample_ids2=None):
    # nummer of padded src frames before and after the video
    n_padded = 4
    # pad width in pxiels for spaceing between different rows of enrollment plots
    pad_width = 10

    # bring in right form
    src1 = scale_imgs(src1).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    src2 = scale_imgs(src2).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    src_with_poke1, _ = make_poke_img(src1, poke1, poke_normalized, poke_coords1)
    src_with_poke2, _ = make_poke_img(src2, poke2, poke_normalized, poke_coords2)
    src_with_poke1 = np.stack(src_with_poke1, axis=0)
    src_with_poke2 = np.stack(src_with_poke2, axis=0)

    first_pad1 = np.stack([src_with_poke1] * n_padded, axis=1)
    first_pad2 = np.stack([src_with_poke2] * n_padded, axis=1)

    vid1 = scale_imgs(vid1).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
    vid2 = scale_imgs(vid2).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
    src1_pad = np.stack([src1] * n_padded, axis=1)
    src2_pad = np.stack([src2] * n_padded, axis=1)
    vid1 = np.concatenate([src1_pad, vid1], axis=1)
    vid2 = np.concatenate([src2_pad, vid2], axis=1)
    vid1_list = make_temporal_border(vid1, poke1, n_logged=vid1.shape[0], draw_endpoint=False, n_pad_frames=n_padded, concat=False)
    if sample_ids1 is not None:
        vid1 = np.concatenate([np.stack([put_text(frame, f'ID: {sid}') for frame in vid], axis=0) for vid, sid in zip(vid1_list, sample_ids1)], axis=2)
    vid2_list = make_temporal_border(vid2, poke2, n_logged=vid1.shape[0], draw_endpoint=False, n_pad_frames=n_padded, concat=False)
    if sample_ids2 is not None:
        vid2 = np.concatenate([np.stack([put_text(frame, f'ID: {sid}') for frame in vid], axis=0) for vid, sid in zip(vid2_list, sample_ids2)], axis=2)
    vid1 = put_text_to_video_row(vid1, f'Motion 1; ID', display_frame_nr=True, n_padded=n_padded)
    vid2 = put_text_to_video_row(vid2, f'Motion 2 ID', display_frame_nr=True, n_padded=n_padded)

    m2_c1 = scale_imgs(m2_c1).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
    m1_c2 = scale_imgs(m1_c2).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)

    m1_c2 = np.concatenate([first_pad2, src2[:, None], m1_c2], axis=1)
    m2_c1 = np.concatenate([first_pad1, src1[:, None], m2_c1], axis=1)
    m1_c2_list = make_temporal_border(m1_c2, poke2, n_logged=m1_c2.shape[0], draw_endpoint=not poke_normalized,
                                      n_pad_frames=n_padded, poke_coords=poke_coords2, concat=False)
    m1_c2 = np.concatenate(m1_c2_list, axis=2)

    m2_c1_list = make_temporal_border(m2_c1, poke1, n_logged=m2_c1.shape[0], draw_endpoint=not poke_normalized,
                                      n_pad_frames=n_padded, poke_coords=poke_coords1, concat=False)
    m2_c1 = np.concatenate(m2_c1_list, axis=2)
    m1_c2 = put_text_to_video_row(m1_c2, 'Transfer: Motion 1, Cond 2', display_frame_nr=True, n_padded=n_padded)
    m2_c1 = put_text_to_video_row(m2_c1, 'Transfer: Motion 2, Cond 1', display_frame_nr=True, n_padded=n_padded)

    complete_grid = np.concatenate([vid1, m1_c2, vid2, m2_c1], axis=1)

    if make_enrollment:
        enrollments = []
        for v1, tm1_c2, v2, tm2_c1 in zip(vid1_list, m1_c2_list, vid2_list, m2_c1_list):
            v1_enroll = np.concatenate(list(v1), axis=1)
            v2_enroll = np.concatenate(list(v2), axis=1)
            tm1_c2_enroll = np.concatenate(list(tm1_c2), axis=1)
            tm2_c1_enroll = np.concatenate(list(tm2_c1), axis=1)
            pad = np.full((pad_width, *v1_enroll.shape[1:]), 255, dtype=np.uint8)
            enrollment_grid = np.concatenate([v1_enroll, pad, tm1_c2_enroll, pad, v2_enroll, pad, tm2_c1_enroll], axis=0)

            enrollments.append(enrollment_grid)

        return complete_grid, enrollments

    return complete_grid


def make_transfer_grids_new(src1, src2, poke1, vid1, m1_c2, m_random_c2, poke_coords1, poke_normalized=False, make_enrollment=False, sample_ids1=None, sample_ids2=None):
    # m_random_c2 contains src2 as start image, poke 1 and random motion

    # nummer of padded src frames before and after the video
    n_padded = 4
    # pad width in pxiels for spaceing between different rows of enrollment plots
    pad_width = 10

    # bring in right form
    src1 = scale_imgs(src1).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    src2 = scale_imgs(src2).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    src1_with_poke1, _ = make_poke_img(src1, poke1, poke_normalized, poke_coords1)
    src2_with_poke1, _ = make_poke_img(src2, poke1, poke_normalized, poke_coords1)
    src1_with_poke1 = np.stack(src1_with_poke1, axis=0)
    src2_with_poke1 = np.stack(src2_with_poke1, axis=0)

    first_pad1 = np.stack([src1_with_poke1] * n_padded, axis=1)
    first_pad2 = np.stack([src2_with_poke1] * n_padded, axis=1)

    vid1 = scale_imgs(vid1).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
    # vid2 = scale_imgs(vid2).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
    src1_pad = np.stack([src1] * n_padded, axis=1)
    # src2_pad = np.stack([src2] * n_padded, axis=1)
    vid1 = np.concatenate([src1_pad, vid1], axis=1)
    # vid2 = np.concatenate([src2_pad, vid2], axis=1)
    vid1_list = make_temporal_border(vid1, poke1, n_logged=vid1.shape[0], draw_endpoint=True, n_pad_frames=n_padded, concat=False)
    if sample_ids1 is not None:
        vid1 = np.concatenate([np.stack([put_text(frame, f'Motion ID: {sid[0]}', loc=(int(frame.shape[1] // 3), int(frame.shape[0] / 6)), color=(0, 255, 0), font_scale=float(frame.shape[0] / 512))
                                         for frame in vid], axis=0) for vid, sid in zip(vid1_list, sample_ids1)], axis=2)

    # vid2_list = make_temporal_border(vid2, poke2, n_logged=vid1.shape[0], draw_endpoint=False,n_pad_frames=n_padded, concat=False)
    # if sample_ids2 is not None:
    #     vid2 = np.concatenate([np.stack([put_text(frame,f'ID: {sid}') for frame in vid],axis=0) for vid, sid in zip(vid2_list,sample_ids2)],axis=2)
    # vid1 = put_text_to_video_row(vid1,f'Motion 1; ID {sample_ids1}', display_frame_nr=True,n_padded=n_padded)
    # vid2 = put_text_to_video_row(vid2,f'Motion 2 ID', display_frame_nr=True,n_padded=n_padded)

    # m2_c1 = scale_imgs(m2_c1).detach().permute(0,1,3,4,2).cpu().numpy().astype(np.uint8)
    m1_c2 = scale_imgs(m1_c2).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)

    m1_c2 = np.concatenate([first_pad2, src2[:, None], m1_c2], axis=1)
    # m2_c1 = np.concatenate([first_pad1, src1[:,None], m2_c1], axis=1)
    m1_c2_list = make_temporal_border(m1_c2, poke1, n_logged=m1_c2.shape[0], draw_endpoint=not poke_normalized,
                                      n_pad_frames=n_padded, poke_coords=poke_coords1, concat=False)
    if sample_ids2 is not None:
        m1_c2 = np.concatenate([np.stack([put_text(frame, f'src ID: {sid[0]}', loc=(int(frame.shape[1] // 3), int(frame.shape[0] / 6)), color=(0, 255, 0), font_scale=float(frame.shape[0] / 512))
                                          for frame in vid], axis=0) for vid, sid in
                                zip(m1_c2_list, sample_ids2)], axis=2)

    mr_c2 = scale_imgs(m_random_c2).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)
    mr_c2 = np.concatenate([first_pad2, src2[:, None], mr_c2], axis=1)
    mr_c2_list = make_temporal_border(mr_c2, poke1, n_logged=m1_c2.shape[0], draw_endpoint=not poke_normalized,
                                      n_pad_frames=n_padded, poke_coords=poke_coords1, concat=False)

    # m1_c2 = np.concatenate(m1_c2_list, axis=2)
    mr_c2 = np.concatenate(mr_c2_list, axis=2)
    # m2_c1_list= make_temporal_border(m2_c1, poke1, n_logged=m2_c1.shape[0], draw_endpoint=not poke_normalized,
    #                              n_pad_frames=n_padded, poke_coords=poke_coords1,concat=False)
    # m2_c1 = np.concatenate(m2_c1_list,axis=2)
    m1_c2 = put_text_to_video_row(m1_c2, 'Transfer: Motion 1', display_frame_nr=True, n_padded=n_padded)
    mr_c2 = put_text_to_video_row(mr_c2, 'Random Motion, Cond 1', display_frame_nr=True, n_padded=n_padded)

    complete_grid = np.concatenate([vid1, m1_c2, mr_c2], axis=1)

    if make_enrollment:
        enrollments = []
        single_videos = []
        for v1, tm1_c2, tmr_c2, in zip(vid1_list, m1_c2_list, mr_c2_list):
            video_column = np.concatenate([v1, tm1_c2, tmr_c2], axis=1)
            single_videos.append(video_column)
            v1_enroll = np.concatenate(list(v1), axis=1)
            tm1_c2_enroll = np.concatenate(list(tm1_c2), axis=1)
            tmr_c2_enroll = np.concatenate(list(tmr_c2), axis=1)
            pad = np.full((pad_width, *v1_enroll.shape[1:]), 255, dtype=np.uint8)
            enrollment_grid = np.concatenate([v1_enroll, pad, tm1_c2_enroll, pad, tmr_c2_enroll], axis=0)

            enrollments.append(enrollment_grid)

        return complete_grid, enrollments, single_videos

    return complete_grid


def make_multipoke_grid(src, multipoke, tgt, samples, poke_normalized=False, image_range="float-1", multipoke_coords=None, poke_in_tgt=True):
    n_padded = 4
    # assert samples.size(0) == 1
    src = scale_imgs(src, input_format=image_range).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    samples = list(scale_imgs(samples, input_format=image_range).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8))
    first_frames_pad = []

    for p, pc in zip(multipoke, multipoke_coords):
        swp, _ = make_poke_img(src, p[None], poke_normalized, pc)
        fp = np.stack(swp * n_padded)
        first_frames_pad.append(fp)

    assert len(first_frames_pad) == len(samples)
    samples = [np.concatenate([fp, src, s], axis=0)[None] for s, fp in zip(samples, first_frames_pad)]

    # make target video with padded frames
    tgt = scale_imgs(tgt, input_format=image_range).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)[0]
    first_tgt_pad = np.concatenate([src] * n_padded)
    tgt = np.concatenate([first_tgt_pad, tgt], axis=0)[None]

    tgt_wo_name = make_temporal_border(tgt, multipoke[0][None], n_logged=1, draw_endpoint=not poke_normalized,
                                       n_pad_frames=n_padded, poke_coords=multipoke_coords[0] if poke_in_tgt else None)
    tgt = put_text_to_video_row(tgt_wo_name, 'GT', display_frame_nr=True, n_padded=n_padded)

    samples_out = []
    samples_wo_name = []
    # grid_samples_unlabeled = []
    for i, (sample, poke, poke_coords) in enumerate(zip(samples, multipoke, multipoke_coords)):
        s_wo_name = make_temporal_border(sample, poke[None], n_logged=1, draw_endpoint=not poke_normalized,
                                         n_pad_frames=n_padded, poke_coords=poke_coords, startframe=src)
        s = put_text_to_video_row(s_wo_name, f'Sample #{i + 1}', display_frame_nr=True, n_padded=n_padded)
        samples_out.append(s)
        samples_wo_name.append(s_wo_name)

    samples_out.insert(0, tgt)
    samples_w_gt = samples_out[:-1]
    n_cols, n_rows = get_divisors(len(samples_w_gt))

    samples_wo_name.insert(0, tgt_wo_name)
    # samples_wo_name = samples_wo_name[:-]

    sgrid = np.concatenate([np.concatenate(samples_w_gt[i * n_cols:(i + 1) * n_cols], axis=2) for i in range(n_rows)], axis=1)
    sgrid_unlabeled = np.concatenate([np.concatenate(samples_wo_name[i * n_cols:(i + 1) * n_cols], axis=2) for i in range(n_rows)], axis=1)

    return samples_wo_name, sgrid, sgrid_unlabeled


def make_samples_and_samplegrid(src, poke, tgt, samples, poke_normalized=False, image_range="float-1", poke_coords=None, poke_in_tgt=True):
    n_padded = 4
    # assert samples.size(0) == 1
    src = scale_imgs(src, input_format=image_range).detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    samples = list(scale_imgs(samples, input_format=image_range).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8))
    src_with_poke, _ = make_poke_img(src, poke, poke_normalized, poke_coords)
    first_pad = np.stack(src_with_poke * n_padded)
    samples = [np.concatenate([first_pad, src, s], axis=0)[None] for s in samples]

    tgt = scale_imgs(tgt, input_format=image_range).detach().permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8)[0]
    first_tgt_pad = np.concatenate([src] * n_padded)
    tgt = np.concatenate([first_tgt_pad, tgt], axis=0)[None]
    tgt_wo_name = make_temporal_border(tgt, poke, n_logged=1, draw_endpoint=not poke_normalized,
                                       n_pad_frames=n_padded, poke_coords=poke_coords if poke_in_tgt else None)
    tgt = put_text_to_video_row(tgt_wo_name, 'GT', display_frame_nr=True, n_padded=n_padded)

    samples_out = []
    samples_wo_name = []
    # grid_samples_unlabeled = []
    for i, sample in enumerate(samples):
        s_wo_name = make_temporal_border(sample, poke, n_logged=1, draw_endpoint=not poke_normalized,
                                         n_pad_frames=n_padded, poke_coords=poke_coords, startframe=src)
        s = put_text_to_video_row(s_wo_name, f'Sample #{i + 1}', display_frame_nr=True, n_padded=n_padded)
        samples_out.append(s)
        samples_wo_name.append(s_wo_name)

    samples_out.insert(0, tgt)
    samples_w_gt = samples_out[:-1]
    n_cols, n_rows = get_divisors(len(samples_w_gt))

    samples_wo_name.insert(0, tgt_wo_name)
    samples_wo_name = samples_wo_name[:-1]

    sgrid = np.concatenate([np.concatenate(samples_w_gt[i * n_cols:(i + 1) * n_cols], axis=2) for i in range(n_rows)], axis=1)
    sgrid_unlabeled = np.concatenate([np.concatenate(samples_wo_name[i * n_cols:(i + 1) * n_cols], axis=2) for i in range(n_rows)], axis=1)

    return samples_wo_name, sgrid, sgrid_unlabeled


def save_video(video, savepath, fps=5):
    assert savepath.endswith('.mp4'), f'Only mp4 videos supported.'
    savepath_pre = savepath.split('.mp4')[0] + '_pre.mp4'
    writer = cv2.VideoWriter(
        savepath_pre,
        cv2.VideoWriter_fourcc(*"MP4V"),
        fps,
        (video.shape[2], video.shape[1]),
    )

    # writer = vio.FFmpegWriter(savename,inputdict=inputdict,outputdict=outputdict)

    for frame in video:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    # ensure that

    try:
        os.system(f"ffmpeg -y -hide_banner -loglevel error -i {savepath_pre} -vcodec libx264 {savepath}")
        os.remove(savepath_pre)
    except Exception() as e:
        print(f'The folowing exception was raised {e}')
        print('Probably, ffmpeg is not installed on your system...')


def make_flow_video(src, poke, pred, tgt, n_logged, flow=None, length_divisor=5, logwandb=True, flow_weights=None, display_frame_nr=False, invert_poke=False):
    """

    :param src: src image
    :param poke: poke, also input to the network
    :param pred: predicted video of the network
    :param tgt: target video the network was trained to reconstruct
    :param n_logged: numvber of logged examples
    :param flow: src flow from which the poke is originating
    :param length_divisor: divisor for the length of the arrow, that's drawn ti visualize the mean direction of the flow within the poke patch
    :param logwandb: whether the output video grid is intended to be logged with wandb or not (in this case the grid channels have to be changed)
    :param flow_weights: Optional weights for the flow which are also displayed if they are not None.
    :return:
    """
    seq_len = tgt.shape[1]

    src = ((src.permute(0, 2, 3, 1).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]

    pokes = vis_flow(poke[:n_logged])
    flows_vis = None
    if flow is not None:
        flows = vis_flow(flow[:n_logged])
        flows_with_rect = []
        for i, (poke_p, flow) in enumerate(zip(pokes, flows)):
            poke_points = np.nonzero(poke_p.any(-1) > 0)
            if poke_points[0].size == 0:
                flows_with_rect.append(np.zeros_like(flow))
            else:
                min_y = np.amin(poke_points[0])
                max_y = np.amax(poke_points[0])
                min_x = np.amin(poke_points[1])
                max_x = np.amax(poke_points[1])
                # draw rect
                flow_with_rect = cv2.rectangle(flow, (min_x, min_y), (max_x, max_y), (255, 255, 255), max(1, int(flow.shape[0] // 64)))
                # flow_with_rect = cv2.UMat.get(cv2.putText(cv2.UMat(flow_with_rect), f"Flow Complete",(int(flow_with_rect.shape[1] // 3), int(5 * flow_with_rect.shape[0] / 6) ), cv2.FONT_HERSHEY_SIMPLEX,
                #                        float(flow_with_rect.shape[0] / 256), (255, 255, 255), int(flow_with_rect.shape[0] / 128)))

                flows_with_rect.append(flow_with_rect)

        flow_cat = np.concatenate(flows_with_rect, axis=1)

        flows_vis = [np.stack([flow_cat] * seq_len, axis=0)]
        flows_vis[0] = put_text_to_video_row(flows_vis[0], "Flow Complete", color=(255, 255, 255))

    if flow_weights is not None:
        flow_weights = flow_weights.cpu().numpy()
        heatmaps = []
        for i, weight in enumerate(flow_weights):
            weight_map = ((weight - weight.min()) / weight.max() * 255.).astype(np.uint8)
            heatmap = cv2.applyColorMap(weight_map, cv2.COLORMAP_HOT)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            heatmaps.append(heatmap)

        heatmaps = np.concatenate(heatmaps, axis=1)
        heatmaps = np.stack([heatmaps] * seq_len, axis=0)
        heatmaps = put_text_to_video_row(heatmaps, "Flow Weights", color=(255, 255, 255))
        if flows_vis is None:
            flows_vis = [heatmaps]
        else:
            flows_vis.insert(0, heatmaps)

    srcs_with_arrow = []
    pokes_with_arrow = []
    if invert_poke:
        srcs_with_arrow_inv = []
        pokes_with_arrow_inv = []
    eps = 1e-6
    for i, (poke_p, src_i) in enumerate(zip(poke[:n_logged], src)):
        poke_points = np.nonzero(pokes[i].any(-1) > 0)
        if poke_points[0].size == 0:
            pokes_with_arrow.append(np.zeros_like(pokes[i]))
            srcs_with_arrow.append(src_i)
        else:
            min_y = np.amin(poke_points[0])
            max_y = np.amax(poke_points[0])
            min_x = np.amin(poke_points[1])
            max_x = np.amax(poke_points[1])
            # plot mean direction of flow in poke region
            avg_flow = np.mean(poke_p[:, min_y:max_y, min_x:max_x].cpu().numpy(), axis=(1, 2))
            arrow_dir = avg_flow / (np.linalg.norm(avg_flow) + eps) * (poke_p.shape[1] / length_divisor)
            if not math.isnan(arrow_dir[0]) or not math.isnan(arrow_dir[1]):
                arrow_start = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
                arrow_end = (arrow_start[0] + int(arrow_dir[0]), arrow_start[1] + int(arrow_dir[1]))
                test = pokes[i]
                # test = cv2.UMat.get(cv2.putText(cv2.UMat(test), f"Poke", (int(test.shape[1] // 3), int(5 * test .shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                #                                           float(test.shape[0] / 256), (255, 255, 255), int(test.shape[0] / 128)))
                pokes_with_arrow.append(cv2.arrowedLine(test, arrow_start, arrow_end, (255, 0, 0), max(int(src_i.shape[0] / 64), 1)))
                srcs_with_arrow.append(cv2.UMat.get(cv2.arrowedLine(cv2.UMat(src_i), arrow_start, arrow_end, (255, 0, 0), max(int(src_i.shape[0] / 64), 1))))
                if invert_poke:
                    arrow_end_inv = (arrow_start[0] - int(arrow_dir[0]), arrow_start[1] - int(arrow_dir[1]))
                    pokes_with_arrow_inv.append(cv2.arrowedLine(test, arrow_start, arrow_end_inv, (0, 255, 0), max(int(src_i.shape[0] / 64), 1)))
                    srcs_with_arrow_inv.append(cv2.UMat.get(cv2.arrowedLine(cv2.UMat(src_i), arrow_start, arrow_end, (0, 255, 0), max(int(src_i.shape[0] / 64), 1))))
            else:
                pokes_with_arrow.append(np.zeros_like(pokes[i]))
                srcs_with_arrow.append(src_i)

    poke = np.concatenate(pokes_with_arrow, axis=1)
    if invert_poke:
        poke_inv = np.concatenate(pokes_with_arrow_inv, axis=1)
        poke = put_text_to_video_row(np.stack([*[poke] * int(math.ceil(float(seq_len) / 2)), *[poke_inv] * int(seq_len / 2)], axis=0), "Pokes", color=(255, 255, 255))
    else:
        poke = put_text_to_video_row(np.stack([poke] * seq_len, axis=0), "Poke", color=(255, 255, 255))

    if flows_vis is None:
        flows_vis = [poke]
    else:
        flows_vis.append(poke)

    srcs = np.concatenate(srcs_with_arrow, axis=1)
    srcs = cv2.UMat.get(cv2.putText(cv2.UMat(srcs), f"Sequence length {seq_len}", (int(srcs.shape[1] // 3), int(srcs.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                    float(srcs.shape[0] / 256), (255, 0, 0), int(srcs.shape[0] / 128)))
    if invert_poke:
        srcs_inv = np.concatenate(srcs_with_arrow_inv, axis=1)
        srcs_inv = cv2.UMat.get(cv2.putText(cv2.UMat(srcs_inv), f"Sequence length {seq_len}", (int(srcs_inv.shape[1] // 3), int(srcs_inv.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                            float(srcs_inv.shape[0] / 256), (255, 0, 0), int(srcs_inv.shape[0] / 128)))
        srcs = np.stack([*[srcs] * int(math.ceil(float(seq_len) / 2)), *[srcs_inv] * int(seq_len / 2)], axis=0)
    else:
        srcs = np.stack([srcs] * seq_len, axis=0)
    srcs = put_text_to_video_row(srcs, "Input Image", display_frame_nr=display_frame_nr)

    pred = ((pred.permute(0, 1, 3, 4, 2).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    pred = np.concatenate(list(pred), axis=2)
    pred = put_text_to_video_row(pred, "Predicted Video", display_frame_nr=display_frame_nr)

    tgt = ((tgt.permute(0, 1, 3, 4, 2).cpu().numpy() + 1.) * 127.5).astype(np.uint8)[:n_logged]
    tgt = np.concatenate(list(tgt), axis=2)
    tgt = put_text_to_video_row(tgt, "Groundtruth Video", display_frame_nr=display_frame_nr)

    full = np.concatenate([srcs, pred, tgt, *flows_vis], axis=1)
    if logwandb:
        full = np.moveaxis(full, [0, 1, 2, 3], [0, 2, 3, 1])

    return full


def fig2data(fig, imsize):
    """

    :param fig: Matplotlib figure
    :param imsize:
    :return:
    """
    canvas = FigureCanvas(fig)

    ax = fig.gca()

    # ax.text(0.0, 0.0, "Test", fontsize=45)
    # ax.axis("off")
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    width, height = imsize
    image = image.reshape((int(height), int(width), -1))
    return image


def make_errorbar_plot(fname, dataframe: pd.DataFrame, xid='frame_id', yid='nn_mse_err', hueid='poke_indicator', varid=None, alpha=.3, title=None):
    # fig, ax = plt.subplots()
    # use default theme
    # fig = plt.Figure()

    sns.set_theme()

    unique_hueids, n_hueid = np.unique(dataframe[hueid], return_counts=True)
    # obtain color palette
    colors = sns.color_palette(n_colors=unique_hueids.shape[0])

    for act_hueid, color in zip(unique_hueids, colors):
        ids = np.flatnonzero(dataframe[hueid] == act_hueid)
        ax = sns.lineplot(x=xid, y=yid, data=dataframe.loc[ids], color=color, markers=True, label=f'{act_hueid} Pokes')  # , hue=hueid)
        # get lower and upper bounds for variances
        if varid is not None:
            lb = dataframe[yid][ids].to_numpy() - .5 * dataframe[varid][ids].to_numpy()
            ub = dataframe[yid][ids].to_numpy() + .5 * dataframe[varid][ids].to_numpy()
            ax.fill_between(dataframe[xid][ids].to_numpy(), lb, ub, alpha=alpha, color=color)

    # title = hueid
    # ax._legend.set_title(title)
    # for t, i in zip(ax._legend.texts, unique_hueids):
    #     t.set_text(str(i))
    ax.legend(loc='upper left')
    ax.set_xlim(left=np.min(dataframe[xid]), right=np.max(dataframe[xid]))
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def make_nn_var_plot(n_pokes, nn_per_frame, savedir, std_per_frame=None, save_csv=True):
    os.makedirs(savedir, exist_ok=True)
    data_dict = {'frame_id': np.arange(nn_per_frame.shape[0]),
                 'nn_mse_err': nn_per_frame,
                 'poke_indicator': [f'{n_pokes} poke vectors'] * nn_per_frame.shape[0]}
    if std_per_frame is not None:
        assert std_per_frame.shape[0] == nn_per_frame.shape[0]
        data_dict.update({'std_per_frame': std_per_frame})

    # make dataset
    df = pd.DataFrame.from_dict(data_dict)
    if save_csv:
        save_name = os.path.join(savedir, f'keypoint_err_data_{n_pokes}_pokes.csv')
        df.to_csv(save_name)

    fig_savename = os.path.join(savedir, f'keypoint_err_{n_pokes}_pokes.pdf')

    make_errorbar_plot(fig_savename, df, varid='std_per_frame' if std_per_frame is not None else None)


# def violin_plot(ax,data,groups,bp=False):
#     '''Create violin plot along an axis'''
#     dist = max(groups) - min(groups)
#     w = min(0.15*max(dist,1.0),0.4)
#     # fixme choose denominator such that it is the globally largest std value
#
#     for d,p in zip(data,groups):
#         k = stats.gaussian_kde(d) #calculates the kernel density
#         m = k.dataset.min() #lower bound of violin
#         M = k.dataset.max() #upper bound of violin
#         x = np.arange(m,M,(M-m)/100.) # support for violin
#         v = k.evaluate(x) #violin profile (density curve)
#         v = v/v.max()*w #scaling the violin to the available space
#         ax.fill_betweenx(x,p,v+p,alpha=0.3)
#         ax.fill_betweenx(x,p,-v+p,alpha=0.3)


def make_two_axes_plot(df: pd.DataFrame, savepath, key_y1='Mean MSE', key_y2='Std', x='Number of Pokes', title='Test title'):
    sns.set_theme()
    sns.set_context('poster', rc={"font.size": 1.})
    df_n_poke = df.groupby(x, as_index=False).mean()

    colors = sns.color_palette(n_colors=4)

    x = df_n_poke[x].to_numpy().astype(int)
    y1 = df_n_poke[f'{key_y1} per Frame'].to_numpy()
    y2 = df_n_poke[f'{key_y2} per Frame'].to_numpy()
    #
    fig, ax1 = plt.subplots()

    l1 = ax1.plot(x, y1, 'd--', color=colors[0], label=key_y1, markersize=10)
    y1_offset = (y1.max() - y1.min()) / 10
    min_y1 = y1.min() - y1_offset
    max_y1 = y1.max() + y1_offset
    y1_ticks = np.linspace(min_y1, max_y1, num=3, endpoint=True)
    ax1.set_ylim((min_y1 - y1_offset, max_y1 + y1_offset))
    ax1.set_yticks(y1_ticks)
    ax1.tick_params(axis='y', color=colors[0])
    ax1.set_yticklabels(y1_ticks, color=colors[0], rotation=90, va='center')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

    # ax.legend(key_y1)
    ax2 = ax1.twinx()
    l2 = ax2.plot(x, y2, 'H:', color=colors[3], label=f'{key_y2}-50s', markersize=10)
    y2_offset = (y2.max() - y2.min()) / 5
    y2_pad = (y2.max() - y2.min()) / 10
    min_y2 = y2.min() - y2_offset
    max_y2 = y2.max() + y2_offset
    y2_ticks = np.linspace(min_y2, max_y2, num=3, endpoint=True)
    ax2.grid(False)
    ax2.set_ylim((min_y2 - y2_pad, max_y2 + y2_pad))
    ax2.set_yticks(y2_ticks)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax2.set_yticklabels(y2_ticks, color=colors[3], rotation=90, va='center')
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax2.tick_params(axis='y', color=colors[3])

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{n}' for n in x])
    ax1.set_xlabel('Number of Pokes')
    ax1.set_title(title)
    # plt.legend()
    # # ax.legend()
    # # rects = rects1 + rects2
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='lower left')
    # fig.tight_layout()
    fig.tight_layout()

    fig.savefig(savepath)
    plt.close()


def make_violin_plot(fname, dataframe, xid='frame_id', yid='nn_mse_err', varid='', hueid='poke_indicator'):
    pass


if __name__ == '__main__':
    from scipy import stats
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='variance_reduction', choices=['variance_reduction', 'kps_acc'], help='mode for the script')
    args = parser.parse_args()

    if args.mode == 'variance_reduction':

        # csv_path = 'logs/second_stage_video/generated/plants-16_10d1-bs20-lr1e-3-bn64-fullseq-mfc32-ss128-mf10-endpoint-np5/metrics/plot_data_50pokes_lpips.csv'
        csv_path = '/export/scratch3/ablattma/poking_inn/second_stage_video/generated/iper-16_10d1-bs96-lr1e-3-bn128-fullseq-ss128-mf10-endpoint-np5-mweight/metrics/plot_data_50pokes_kps.csv'
        key = 'MSE'

        df = pd.read_csv(csv_path, sep=',')

        os.makedirs('logs/plots_results', exist_ok=True)
        save_path = 'logs/plots_results/time_plot_iper.pdf'

        make_errorbar_plot(save_path, df, xid='Time', yid=f'Mean {key} per Frame',
                           hueid='Number of Pokes', varid='Std per Frame',
                           title=f'Test {key} plot')

        # colors = sns.color_palette(n_colors=5)
        # fig, ax = plt.subplots()
        df = pd.read_csv(csv_path)

        savepath = '/export/scratch3/ablattma/poking_inn/plots_results/uncertainty_reduction_iper.pdf'
        make_two_axes_plot(df, savepath, key_y1=f'Mean {key}', title='iPER Dataset')

    else:

        in_path = '/export/scratch3/ablattma/poking_inn/plots_results/kps_acc_targeted_ipoke.csv'
        data = pd.read_csv(in_path)
        z_scores = stats.zscore(data['Mean Squared KP Error'])

        abs_z_scores = np.abs(z_scores)

        filtered_entries = (abs_z_scores < 3)  # .all(axis=1)

        data_filtered = data[filtered_entries]

        Means = data_filtered.groupby('Method')['Mean Squared KP Error'].mean()
        # Modes = data_filtered.groupby('Method')['Value'].agg(lambda x: pd.Series.mode(x)[0])

        sns.set_theme()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax = sns.violinplot(x='Method', y='Mean Squared KP Error', data=data_filtered, order=['Hao', 'Hao w/ KP', 'iPOKE (Ours)'], cut=0., inner=None, ax=ax)
        plt.setp(ax.collections, alpha=.9)
        plt.scatter(x=range(len(Means)), y=Means, label=[str(m) for m in Means], c='k')
        plt.ylim(top=0.04)
        # ax = plt.gca()
        # offset = .1
        # for i, txt in enumerate(Means):
        #     ax.text(i+offset,txt-offset,str(txt))

        for index, mean in enumerate(Means):
            ax.text(x=index + .055, y=mean + .0012, s="%.4f" % mean) #, fontdict=dict(fontsize=18))

        # plt.scatter(x=range(len(Modes)), y=Modes)

        plt.tight_layout()
        os.makedirs('logs/plots_result',exist_ok=True)
        fig.savefig('logs/plots_result/kps_acc_violin_squeeze_ipoke.pdf')
        plt.close()

        sns.set_theme()
        fig, ax = plt.subplots()
        ax = sns.boxplot(x='Method', y='Mean Squared KP Error', data=data_filtered, order=['Hao', 'Hao w/ KP', 'Ours'], ax=ax)

        plt.tight_layout()
        plt.savefig('logs/plots_result/kps_acc_box.pdf')
        plt.close()