import glob
import os
import time
import numpy as np
import pandas as pd
from read_utils import get_mapping, read_gt_label, estimate_cost_matrix, avg_gt_activity_datasets
from scipy.optimize import linear_sum_assignment
from reduce_classes import req_numclust
from reduce_classes import *
import argparse
import json
import numpy as np
import math
from sklearn import metrics
# import torch
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.metrics.pairwise import cosine_distances

import math

import copy

# 特征归一化函数
from sklearn import preprocessing

from scipy.spatial.distance import cosine


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


def divide_class(features, varepsilon):
    n_frames = len(features)
    req = np.zeros(n_frames, dtype="int32")
    dis = []
    step_dist = 5
    dis_step = metrics.pairwise.pairwise_distances(
        features, features, metric='cosine')
    for i in range(n_frames - step_dist):
        dis.append(dis_step[i][i+step_dist])

    bound = np.where(dis == np.max(
        dis[int(n_frames*varepsilon):int(n_frames*(1-varepsilon))]))
    if(len(bound) == 1):
        bound = bound[0]
    if(len(bound) > 1):
        bound = bound[0]
    for j in range(n_frames):
        if(j > int(bound)):
            req[j] += 1
    return req


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


def run_TDC(
        dataset_name='50Salads',
        datasets_path="./dataset",
        alpha=None,
        beta=None,
        save_arr=True,
):
    # setup paths
    ds_name = dataset_name
    path_ds = os.path.join(datasets_path, ds_name)
    path_gt = os.path.join(path_ds, 'groundTruth/')
    path_mapping = os.path.join(path_ds, 'mapping', 'mapping.txt')
    path_features = os.path.join(path_ds, 'features/')

    # %% Load all needed files: Descriptor, GT & Mapping
    if os.path.exists(path_mapping):
        # Create the Mapping dict
        mapping_dict = get_mapping(path_mapping)
    else:
        mapping_dict = None

    # Load all filenames from ds path
    if ds_name == "MPII_Cooking":  # Load MPI test set files，这个数据集特别一点
        mpi_df = pd.read_csv(os.path.join(
            path_ds, 'mapping', 'sequencesTest.txt'), names=['filename'], header=None)
        mpi_df['filename'] = path_features + \
            mpi_df['filename'] + '-cam-002.txt'
        filenames = mpi_df.filename.tolist()
    else:
        filenames = glob.glob(os.path.join(
            path_features, '**/*.txt'), recursive=True)

    # Create the result matrix to hold all values in the end
    results_matrix_ds = np.zeros(shape=(len(filenames), 4), dtype=object)

    for file_num, cur_filename in enumerate(filenames):

        video_name = os.path.basename(cur_filename)[:-4]
        if ds_name == "MPII_Cooking":
            activity_name = video_name.split("-")[1]
        else:
            activity_name = os.path.basename(os.path.split(cur_filename)[0])

        # n_clusters for TDC
        n_clusters = int(avg_gt_activity_datasets[ds_name][activity_name])

        cur_desc = np.loadtxt(cur_filename, dtype='float32')

        length_video = len(cur_desc)

        window = int(alpha * (length_video / n_clusters))
        if(window % 2 == 0):
            window += 1

        # moving average
        use_smoothing = True
        if(use_smoothing):
            for i in range(len(cur_desc)):
                if(i < window/2):
                    cur_desc[i] = np.mean(
                        cur_desc[:i + int(window/2)+1], axis=0)
                elif(i >= len(cur_desc) - window/2):
                    cur_desc[i] = np.mean(
                        cur_desc[i - int((window)/2):], axis=0)
                else:
                    cur_desc[i] = np.mean(
                        cur_desc[i - int((window)/2):i + int(window/2)+1], axis=0)

        save_averaged_features = True
        if(save_averaged_features):
            save_averaged_features_path = os.path.join(
                datasets_path, ds_name,  "features_average")
            np.save(os.path.join(save_averaged_features_path,
                                 cur_filename.split('/')[-1][:-4] + ".npy"), cur_desc)

        if(ds_name == "MPII_Cooking"):
            v = 0.2
        else:
            v = 0.1

        req_c = divide_class(cur_desc, v)
        num_class = np.unique(req_c)
        while (len(num_class) < int(n_clusters*beta)):
            distance = []
            for i in range(len(num_class)):
                sum_dist = 0
                loc = np.where(req_c == i)
                if (len(loc) == 1):
                    loc = loc[0]
                avg = np.mean(cur_desc[loc], axis=0)
                avg = avg[np.newaxis, ...]
                distance1 = metrics.pairwise.pairwise_distances(
                    cur_desc[loc], avg, metric='cosine')
                for j in range(len(loc)):
                    sum_dist += distance1[j][0]**2
                distance.append(sum_dist)

            max_num = np.where(distance == np.max(distance))
            max_num = max_num[0]
            if (len(max_num) > 1):
                max_num = max_num[0]
            location = np.where(req_c == max_num)

            for i in range(len(req_c)):
                if (req_c[i] > int(max_num)):
                    req_c[i] += 1

            req = divide_class(cur_desc[location[0]], v)
            req_c[location[0]] = req + int(max_num)

            num_class = np.unique(req_c)

        n_frames = len(cur_desc)
        time_index = (np.arange(n_frames) + 1.) / n_frames
        cur_desc = np.concatenate(
            [cur_desc, time_index[..., np.newaxis]], axis=1)

        if(ds_name == "MPII_Cooking"):
            req_c = req_numclust_notime_weighted(req_c, cur_desc, req_clust=n_clusters,
                                                 distance='cosine')
        else:
            req_c = req_numclust(req_c, cur_desc, req_clust=n_clusters,
                                 distance='cosine', use_tw_finch=True)
        req_c = req_c[0]

        gt_label_path = os.path.join(path_gt, video_name)  # get groundtruth
        if not os.path.exists(gt_label_path):
            gt_label_path = os.path.join(path_gt, video_name + '.txt')

        gt_labels, n_labels = read_gt_label(
            gt_label_path, mapping_dict=mapping_dict)

        cost_matrix = estimate_cost_matrix(gt_labels, req_c)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        y_pred = col_ind[req_c]  #

        Salads_eval = False
        if Salads_eval and ds_name == "50Salads":
            y_pred = fs_eval_mode_map(y_pred, datasets_path)
            gt_labels = fs_eval_mode_map(gt_labels, datasets_path)
            n_clusters = len(np.unique(y_pred))

        cur_acc = metrics.accuracy_score(gt_labels, y_pred)

        f1_macro = metrics.f1_score(gt_labels, y_pred, average='macro')
        iou = np.sum(metrics.jaccard_score(
            gt_labels, y_pred, average=None)) / n_clusters

        # Save .npy for training SS-BRN
        # Save the .txt file for calculating scores using MATLAB
        if(save_arr):
            save_req_path = os.path.join(datasets_path, ds_name, "req_TDC")
            save_clustering_path = os.path.join(
                datasets_path, ds_name, "cluster_arr")
            if not os.path.exists(save_req_path):
                os.mkdir(save_req_path)
            if not os.path.exists(save_clustering_path):
                os.mkdir(save_clustering_path)
            np.savetxt(os.path.join(save_req_path,
                       cur_filename.split('/')[-1][:-4] + ".txt"), req_c)
            np.save(os.path.join(save_clustering_path,
                    cur_filename.split('/')[-1][:-4] + ".npy"), req_c)

        print(
            f'Evaluation on Video {activity_name}/{video_name} : accuracy = {cur_acc} IoU = {iou} and f1 ={f1_macro}')

        # Transfer all the calculated metrics into the result matrix
        results_matrix_ds[file_num][0] = activity_name + '/' + video_name
        results_matrix_ds[file_num][1] = cur_acc  # Mean over frame accuracy
        results_matrix_ds[file_num][2] = iou  # IOU
        results_matrix_ds[file_num][3] = f1_macro  # F1 from Sklearn

    avg_score = np.mean(results_matrix_ds[:, 1:4], axis=0)
    print(
        f'Overal Results on {ds_name} Dataset: MOF:{avg_score[0]}, IOU: {avg_score[1]}, F-Score: {avg_score[2]}')
    return avg_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='50Salads',
                        help='Specify the path to your data csv file.')
    parser.add_argument('--datasets-path', default="./dataset",
                        help='Specify the root folder of all datsets')
    parser.add_argument('--alpha', default=0.11)
    parser.add_argument('--beta', default=1.0)
    parser.add_argument('--Save_req',  default=True)
    # args = parser.parse_args()
    args = parser.parse_args(args=[])
    # args, unknown = parser.parse_known_args()
    _ = run_TDC(dataset_name=args.dataset_name,
                datasets_path=args.datasets_path,
                tw_finch=args.tw_finch,
                verbose=args.verbose)
