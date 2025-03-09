import argparse
import glob
import os
from typing import List

import numpy as np
from PIL import Image


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="convert np.array to images.")
    parser.add_argument(
        "dir",
        type=str,
        help="/share/liuyule/Action_Segmentation_Datasets/50Salads/req_TW/rgb-13-2.txt",
    )

    return parser.parse_args()


def convert_arr2img(arr: np.ndarray, palette: List[int]) -> Image.Image:
    """
    Args:
        arr: 1d array(T, )
        palette: color palette
    """
    arr = arr.astype(np.uint8)
    arr = np.tile(arr, (100, 1))
    img = Image.fromarray(arr)
    img = img.convert("P")
    img.putpalette(palette)

    return img


def main() -> None:
    # args = get_arguments()

    voc = Image.open("./imgs/voc_sample.png")
    voc = voc.convert("P")
    palette = voc.getpalette()

    # arr_paths = glob.glob(os.path.join(args.dir, "*.npy"))

    # arr_paths = glob.glob(os.path.join(
    #     "/share/liuyule/Action_Segmentation_Datasets/50Salads/req_TW/rgb-19-1.txt"))

    arr_paths = glob.glob(os.path.join(
        "/share/liuyule/Action_Segmentation_Datasets/50Salads/gt_arr_groundTruth_txt/rgb-11-2.txt"))

    for path in arr_paths:
        name = os.path.basename(path)[:-4]  # remove .npy
        # arr = np.load(path)

        arr = np.loadtxt(path, dtype="float64")

        img = convert_arr2img(arr, palette)
        img.save(os.path.join(
            "/home/liuyule/asrf-main-50Salads/imgs", "groundTruth_rgb-11-2" + ".png"))

    print("Done")

    # for i in range(18):

    #     j = str(i+1)

    #     arr_paths = glob.glob(os.path.join(
    #         "/share/liuyule/Action_Segmentation_Datasets/50Salads/pictures/" + j + "_breakfast.txt"))
    #     arr_paths = glob.glob(os.path.join(
    #         "/share/liuyule/Action_Segmentation_Datasets/Breakfast/req_TW/P12_webcam01_P12_cereals.txt"))

    #     arr_paths = glob.glob(os.path.join(
    #         "/share/liuyule/Action_Segmentation_Datasets/50Salads/req_TW/P12_webcam01_P12_cereals.txt"))

    #     for path in arr_paths:
    #         name = os.path.basename(path)[:-4]  # remove .npy
    #         # arr = np.load(path)

    #         arr = np.loadtxt(path, dtype="float64")

    #         img = convert_arr2img(arr, palette)
    #         img.save(os.path.join(
    #             "/home/liuyule/asrf-main-50Salads/br",  str(100) + ".png"))

    #     print("Done")


if __name__ == "__main__":
    main()
