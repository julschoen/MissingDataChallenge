import argparse
from skimage import io
import os
import pathlib
import numpy as np
from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list
import numpy as np
from skimage.transform import rotate

def augment(im, mask):
    shift = np.random.randint(1,11)
    ax = np.random.randint(0,2)
    deg = np.random.randint(1,6)

    if np.random.uniform() > 0.5:
        im = np.roll(im, shift, ax)
        mask = np.roll(mask, shift, ax)

    if np.random.uniform() > 0.5:
        im = rotate(im, deg)
        mask = rotate(mask, deg)

    return im, mask

def train_in_painter(settings):
    """
    Computes an average image based on all images in the training set.
    """
    input_data_dir = settings["dirs"]["input_data_dir"]
    output_data_dir = settings["dirs"]["output_data_dir"]
    training_set = settings["data_set"] + ".txt"
    output_dir = os.path.join(output_data_dir, "trained_model")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f'Training inpainter with data set: {training_set} and placing model in {output_dir}')

    file_list = os.path.join(input_data_dir, "data_splits", training_set)
    file_ids = read_file_list(file_list)
    if file_ids is None:
        return

    data_set = settings["data_set"]
    inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")
    pathlib.Path(inpainted_result_dir).mkdir(parents=True, exist_ok=True)

    print(f"Training on {len(file_ids)} images")

    sum_image = None
    for idx in file_ids:
        in_image_name = os.path.join(input_data_dir, "masked", f"{idx}_stroke_masked.png")
        in_mask_name = os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png")
        out_image_name = os.path.join(inpainted_result_dir, f"{idx}.png")

        im = io.imread(in_image_name)
        mask = io.imread(in_mask_name)

        mask = mask/255

        im_ = np.fliplr(im).copy()
        mask_ = np.fliplr(mask).copy()

        im[np.where(mask == 1)] = im_[np.where(mask==1)]

        mask = np.logical_xor(mask, mask_)

        for i in range(1000):
            im_ = np.fliplr(im).copy()
            mask_ = np.fliplr(mask).copy()

            im_, mask_ = augment(im_, mask_)

            im[np.where(mask == 1)] = im_[np.where(mask==1)]

            mask = np.logical_xor(mask, mask_)

            if (mask == 0).all():
                break

        io.imsave(out_image_name, im)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='TrainInPainter')
    config = InPaintConfig(args)
    if config.settings is not None:
        train_in_painter(config.settings)
