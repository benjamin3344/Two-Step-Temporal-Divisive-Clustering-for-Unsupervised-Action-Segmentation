# from libs.class_id_map import get_class2id_map
import argparse
import glob
import os
import sys

import numpy as np

from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

dataset_names = ["50Salads", "breakfast", "gtea"]


def get_class2id_map(dataset: str, dataset_dir: str = "./dataset") -> Dict[str, int]:
    """
    Args:
        dataset: 50salads, gtea, breakfast
        dataset_dir: the path to the datset directory
    """

    assert (
        dataset in dataset_names
    ), "You have to choose 50salads, gtea or breakfast as dataset."

    with open(os.path.join(dataset_dir, "{}/mapping.txt".format(dataset)), "r") as f:
        actions = f.read().split("\n")[:-1]

    class2id_map = dict()
    for a in actions:
        class2id_map[a.split()[1]] = int(a.split()[0])

    return class2id_map


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert ground truth txt files to numpy array"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/share/Segmention/dataset/",
        help="path to a dataset directory (default: ./dataset)",
    )

    return parser.parse_args()


def main() -> None:
    # args = get_arguments()

    # datasets = ["50salads", "gtea", "breakfast"]
    datasets = ["50Salads"]

    for dataset in datasets:
        # make directory for saving ground truth numpy arrays
        save_dir = os.path.join(
            "/share/liuyule/Action_Segmentation_Datasets/", dataset, "gt_arr_groundTruth_txt")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # class to index mapping
        class2id_map = get_class2id_map(
            dataset, "/share/liuyule/Action_Segmentation_Datasets/")

        gt_dir = os.path.join("/share/liuyule/Action_Segmentation_Datasets/",
                              dataset, "groundTruth")
        gt_paths = glob.glob(os.path.join(gt_dir, "*.txt"))

        for gt_path in gt_paths:
            # the name of ground truth text file
            gt_name = os.path.relpath(gt_path, gt_dir)

            with open(gt_path, "r") as f:
                gt = f.read().split("\n")[:-1]

            gt_array = np.zeros(len(gt))
            for i in range(len(gt)):
                gt_array[i] = class2id_map[gt[i]]
                x = gt_array[i]
                n = type(x)
                n = 1

            # save array
            # np.save(os.path.join(save_dir, gt_name[:-4] + ".npy"), gt_array)
            np.savetxt(os.path.join(save_dir, gt_name[:-4] + ".txt"), gt_array)

    print("Done")


if __name__ == "__main__":
    main()
