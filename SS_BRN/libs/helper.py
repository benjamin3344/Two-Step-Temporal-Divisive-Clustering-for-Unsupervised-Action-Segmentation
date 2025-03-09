import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from libs.class_id_map import get_id2class_map
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from libs.postprocess import PostProcessor

import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

from libs.twfinch import *
from libs.read_utils import avg_gt_activity_datasets

import scipy as sio

import copy
# 特征归一化函数
from sklearn import preprocessing

import pandas as pd


def fs_eval_mode_map(labs, datasets_path):
    path_mapping = os.path.join(
        datasets_path, '50Salads', 'mapping', 'mapping.txt')
    df = pd.read_csv(path_mapping, sep=" ", header=None,
                     names=['index', 'action_name'])
    mapping = dict(zip(df['index'], df['action_name']))
    action = [mapping[i] for i in labs]

    path_mapping = os.path.join(
        datasets_path, '50Salads', 'mapping', 'mappingeval.txt')
    df = pd.read_csv(path_mapping, sep=" ", header=None,
                     names=['index', 'action_name'])
    mapping = dict(zip(df['action_name'], df['index']))
    labels = [mapping[i] for i in action]
    labels = np.array(labels)

    return labels


def divide_class(features):
    n_frames = len(features)
    req = np.zeros(n_frames, dtype="int32")
    dis = []
    step_dist = 1
    dis_step = metrics.pairwise.pairwise_distances(
        features, features, metric='cosine')
    for i in range(n_frames - step_dist):
        dis.append(dis_step[i][i+step_dist])
    # bound = np.where(dis == np.max(dis[1:-1]))
    bound = np.where(dis == np.max(dis[int(n_frames*0.1):int(n_frames*0.9)]))
    if(len(bound) == 1):
        bound = bound[0]
    if(len(bound) > 1):
        bound = bound[0]
    for j in range(n_frames):
        if(j > int(bound)):
            req[j] += 1
    return req


def distance_req(cur_desc, req_c):
    num_class = np.unique(req_c)
    distance = []
    for i in range(len(num_class)):
        sum_dist = 0
        loc = np.where(req_c == i)
        if(len(loc) == 1):
            loc = loc[0]
        avg = np.mean(cur_desc[loc], axis=0)
        avg = avg[np.newaxis, ...]
        distance1 = metrics.pairwise.pairwise_distances(
            cur_desc[loc], avg, metric='cosine')
        # distance1 = metrics.pairwise.euclidean_distances(cur_desc[loc], avg)
        for j in range(len(loc)):
            sum_dist += distance1[j][0]**2
            # sum_dist += distance1[j][0]
        # distance.append(sum_dist/len(loc))
        distance.append(sum_dist)
    return sum(distance)


def distance_class(cur_desc, req_c, class_num):
    req_new = copy.deepcopy(req_c)
    location = np.where(req_new == class_num)
    if(len(location) == 1):
        location = location[0]
    if(len(location) == 1):  # 如果这个类只有一个，则直接无视不能分割，返回一个da'zhi
        return 1000000

    req = divide_class(cur_desc[location])

    for i in range(len(req_new)):
        if(req_new[i] > int(class_num)):
            req_new[i] += 1

    req_new[location] = req + int(class_num)

    distance = distance_req(cur_desc=cur_desc, req_c=req_new)

    return distance


def estimate_cost_matrix(gt_labels, cluster_labels):
    # Make sure the lengths of the inputs match:
    if len(gt_labels) != len(cluster_labels):
        print('The dimensions of the gt_labls and the pred_labels do not match')
        return -1
    L_gt = np.unique(gt_labels)
    L_pred = np.unique(cluster_labels)
    nClass_pred = len(L_pred)
    dim_1 = max(nClass_pred, np.max(L_gt) + 1)
    profit_mat = np.zeros((nClass_pred, dim_1))
    for i in L_pred:
        idx = np.where(cluster_labels == i)
        gt_selected = gt_labels[idx]
        n, indices = np.unique(gt_selected, return_index=True)
        for j in L_gt:
            m = (gt_selected == j)
            profit_mat[i][j] = np.count_nonzero(
                gt_selected == j)
    return -profit_mat


def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    optimizer: optim.Optimizer,
    epoch: int,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    # switch training mode
    model.train()

    for i, sample in enumerate(train_loader):
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]

        x = x.to(device)
        t = t.to(device)
        b = b.to(device)
        mask = mask.to(device)

        batch_size = x.shape[0]

        # compute output and loss
        output_bound = model(x)

        loss = 0.0

        if isinstance(output_bound, list):
            n = len(output_bound)
            for out in output_bound:
                loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
        else:
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

        # record loss
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def compute_clustScores(label_gt, y_pred):
    confusionMat = metrics.confusion_matrix(label_gt, y_pred)

    acc = np.trace(confusionMat)/sum(confusionMat)

    recall = np.diag(confusionMat)/sum(confusionMat, 1)

    precision = np.diag(confusionMat)/sum(confusionMat, 0)

    f1Scores = 2*(precision*recall)/(precision + recall)

    precision[np.isnan(precision)] = 0
    recall[np.isnan(recall)] = 0
    f1Scores[np.isnan(f1Scores)] = 0

    precision = np.mean(precision)
    recall = np.mean(recall)
    fscore = np.mean(f1Scores)

    return fscore


def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    boundary_th: float,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(
        refinement_method, boundary_th)  # 这个地方应该是使用了投票，投票的地方需要改

    # switch to evaluate mode
    model.eval()

    results_matrix_ds = np.zeros(shape=(len(val_loader), 7), dtype=object)
    f1_score_refined = []

    file_num = 0

    with torch.no_grad():
        for sample in val_loader:  #

            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            gt = sample["groundTruth"]
            gt = gt.to(device)

            feature_path = sample["feature_path"][0]

            # compute output and loss
            output_bound = model(x)

            # calcualte accuracy and f1 score
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            gt = gt.to("cpu").data.numpy()

            refined_output_cls, boundarys = postprocessor(
                t, boundaries=output_bound, masks=mask
            )

            boundarys = boundarys[0][0]

            ds_name = feature_path.split('/')[2]

            # Save the .txt file for calculating scores using MATLAB
            save_req = False
            if(save_req):
                datasets_path = "./dataset"
                save_req_path = os.path.join(
                    datasets_path, ds_name, "/req_SSBRN")
                if not os.path.exit(save_req_path):
                    os.mkdir(save_req_path)
                np.savetxt(os.path.join(save_req_path,
                           feature_path.split('/')[-1][:-4] + ".txt"),  refined_output_cls[0])

            while(len(np.unique(refined_output_cls[0])) != np.max(refined_output_cls[0])+1):
                for i in range(np.max(refined_output_cls[0])):
                    list_cls = np.where(refined_output_cls[0] == i)
                    if len(list_cls[0]) == 0:
                        for j in range(len(refined_output_cls[0])):
                            if refined_output_cls[0][j] > i:
                                refined_output_cls[0][j] -= 1

            num_class_refined = np.unique(
                refined_output_cls, return_counts=True)
            if(ds_name == "50Salads"):
                activity_name = str(feature_path.split('/')[-2])
            elif(ds_name == "Breakfast"):
                activity_name = str(feature_path.split('/')
                                    [-1]).split('_')[-1][:-4]
            elif(ds_name == "MPII_Cooking"):
                activity_name = feature_path.split(
                    '/')[-1].split('_')[-1][:-4]
                activity_name = activity_name.split('-')[1]
            n_clusters = int(avg_gt_activity_datasets[ds_name][activity_name])

            x = x[0].T

            if(len(refined_output_cls) == 1):
                refined_output_cls = refined_output_cls[0]

            if(len(num_class_refined[0]) < n_clusters):

                while(len(num_class_refined[0]) < n_clusters):
                    distance = []
                    for i in range(len(num_class_refined[0])):
                        sum_dist = 0
                        loc = np.where(refined_output_cls == i)
                        if(len(loc) == 1):
                            loc = loc[0]
                        avg = np.mean(x[loc], axis=0)
                        avg = avg[np.newaxis, ...]
                        distance1 = metrics.pairwise.pairwise_distances(
                            x[loc], avg, metric='cosine')
                        for j in range(len(loc)):
                            sum_dist += distance1[j][0]**2
                            # sum_dist += distance1[j][0]
                        distance.append(sum_dist/len(loc))
                        # distance.append(sum_dist)

                    max_num = np.where(distance == np.max(distance))
                    max_num = max_num[0]
                    if(len(max_num) > 1):
                        max_num = max_num[0]
                    location = np.where(refined_output_cls == max_num)

                    for i in range(len(refined_output_cls)):
                        if(refined_output_cls[i] > int(max_num)):
                            refined_output_cls[i] += 1

                    num_class_refined = np.unique(
                        refined_output_cls, return_counts=True)

                    use_divide_class = True
                    if(use_divide_class):
                        dis = []
                        step_dist = 5
                        dis_step = metrics.pairwise.pairwise_distances(
                            x[location[0]], x[location[0]], metric='cosine')

                        for i in range(len(location[0]) - step_dist):
                            dis.append(
                                dis_step[i][i+step_dist] * (location[0][i+step_dist] - location[0][i]))

                        if(len(location[0]) >= 3):
                            bound = np.where(dis == np.max(dis[1:-1]))
                        else:
                            bound = np.where(dis == np.max(dis))
                        bound = bound[0][int(len(bound)/2)]

                        for j in range(len(location[0])):
                            if(j > int(bound)):
                                refined_output_cls[location[0][j]] += 1

                    num_class_refined = np.unique(
                        refined_output_cls, return_counts=True)

            while(len(np.unique(refined_output_cls)) != np.max(refined_output_cls)+1):
                for i in range(np.max(refined_output_cls)):
                    list_cls = np.where(refined_output_cls == i)
                    if len(list_cls[0]) == 0:
                        for j in range(len(refined_output_cls)):
                            if refined_output_cls[j] > i:
                                refined_output_cls[j] -= 1

            if(refined_output_cls.shape[0] == 1):
                refined_output_cls = refined_output_cls[0]

            save_req = False
            if(save_req):
                np.savetxt("/share/liuyule/Action_Segmentation_Datasets/50Salads/req_asrf/" +
                           feature_path.split('/')[-1][:-4] + ".txt",  refined_output_cls)

            save_feature = False
            if(save_feature):
                out = out[0]
                out = out.T
                np.savetxt("/share/liuyule/Action_Segmentation_Datasets/50Salads/features_tcn/" +
                           feature_path.split('/')[-1][:-4] + ".txt",  out)

            cost_matrix = estimate_cost_matrix(gt[0], refined_output_cls)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            refined_output_cls = col_ind[refined_output_cls]

            refined_output_cls = refined_output_cls[np.newaxis, :]

            # save_req = False
            # if(save_req):
            #     np.savetxt("/share/liuyule/Action_Segmentation_Datasets/50Salads/asrf/" +
            #                feature_path.split('/')[-1][:-4] + ".txt",  refined_output_cls[0])

            Salads_eval = False
            if Salads_eval:
                refined_output_cls[0] = fs_eval_mode_map(
                    refined_output_cls[0], "/share/liuyule/Action_Segmentation_Datasets/")
                gt[0] = fs_eval_mode_map(
                    gt[0], "/share/liuyule/Action_Segmentation_Datasets/")
                n_clusters = len(np.unique(refined_output_cls[0]))

            num_class_refined = np.unique(
                refined_output_cls, return_counts=True)
            if(len(num_class_refined[0]) != n_clusters):
                print("after:", len(num_class_refined[0]))

            cur_acc = metrics.accuracy_score(gt[0], refined_output_cls[0])

            iou = np.sum(metrics.jaccard_score(
                gt[0], refined_output_cls[0], average=None)) / n_clusters

            f1_macro = metrics.f1_score(
                gt[0], refined_output_cls[0], average='macro')
            f1_score_refined.append(f1_macro)

            results_matrix_ds[file_num][0] = cur_acc
            results_matrix_ds[file_num][1] = iou
            results_matrix_ds[file_num][2] = f1_macro

            video_name = str(feature_path.split('/')[-1])

            # print(
            #     f'Evaluation on Video {video_name}  + num_class:{len(np.unique(gt[0]))}: accuracy = {cur_acc} IoU = {iou} and f1 ={f1_macro}')

            # num_gt = len(np.unique(gt))
            # print(
            #     f'Evaluation on Video {video_name}  seconds: gt/videos = {num_gt}')

            file_num += 1

    avg_score = np.mean(results_matrix_ds[:, 0:3], axis=0)
    print(
        f'Overal Results on 50Salads Dataset: MOF:{avg_score[0]}, IOU: {avg_score[1]}, F-Score: {avg_score[2]}')
