import argparse
from skimage import io
import os
import pathlib
import numpy as np
from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list
import numpy as np
from skimage.transform import rotate
from skimage.filters import gaussian


def augment(x, y):
    x_, y_ = x.copy(), y.copy()
    shift = np.random.randint(1,11)
    ax = np.random.randint(0,2)
    deg = np.random.randint(1,6)

    if np.random.uniform() > 0.5:
        x_ = np.roll(x_, shift, ax)
        y_ = np.roll(y_, shift, ax)
    else:
        x_ = rotate(x_, deg, preserve_range=True)
        y_ = rotate(y_, deg, cval=1, preserve_range=True)

    return x_, y_

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

        im_ = im.copy()
        mask_ = mask.copy()
        im_ = np.fliplr(im_)
        mask_ = np.fliplr(mask_)

        and_mask = np.logical_and(mask, np.logical_not(mask_))
        im[and_mask] = im_[and_mask]
        mask[and_mask] = 0
        
        for i in range(1000):
            im_ = im.copy()
            mask_ = mask.copy()
            im_ = np.fliplr(im_)
            mask_ = np.fliplr(mask_)

            im_aug, mask_aug = augment(im_, mask_)

            and_mask = np.logical_and(mask, np.logical_not(mask_aug))
            im[and_mask] = im_aug[and_mask]
            mask[and_mask] = 0

            if (mask == 0).all():
                break

        im = gaussian(im, 1, mode='reflect', preserve_range=True).astype(np.uint8)
        io.imsave(out_image_name, im)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='TrainInPainter')
    config = InPaintConfig(args)
    if config.settings is not None:
        train_in_painter(config.settings)
