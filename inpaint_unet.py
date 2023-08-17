import argparse
from skimage import io
import os
import pathlib
import numpy as np
from inpaint_config import InPaintConfig
from inpaint_tools import read_file_list
from tqdm import tqdm
from unet_model import UNet
import torch


def inpaint_one_image(model, in_image, mask):
    in_image = in_image.astype(np.float32)/255
    in_image = (in_image*2)-1
    mask = mask.astype(np.float32)
    in_image = torch.from_numpy(in_image).float().cuda().permute(2,0,1)
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(0)

    in_image = torch.concat((in_image, mask), dim=0)
    with torch.no_grad():
        rec = model(in_image.unsqueeze(0)).squeeze()

    in_image = in_image[:3]
    mask = mask/255
    rec[torch.where(mask == 0)] = in_image[torch.where(mask == 0)] 
    rec = rec.detach().cpu().permute(1,2,0).numpy()
    rec = (rec+1)/2
    rec = rec*255
    rec = rec.astype(np.uint8)



    return rec


def inpaint_images(settings):
    input_data_dir = settings["dirs"]["input_data_dir"]
    output_data_dir = settings["dirs"]["output_data_dir"]
    data_set = settings["data_set"]
    model_dir = os.path.join(output_data_dir, "trained_model")

    model = UNet().cuda()
    checkpoint = os.path.join("unet_l1_orig", "models", 'checkpoint.pt')
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])


    inpainted_result_dir = os.path.join(output_data_dir, f"inpainted_{data_set}")
    pathlib.Path(inpainted_result_dir).mkdir(parents=True, exist_ok=True)

    print(f"InPainting {data_set} and placing results in {inpainted_result_dir} with model from {model_dir}")


    file_list = os.path.join(input_data_dir, "data_splits", data_set + ".txt")
    file_ids = read_file_list(file_list)
    if file_ids is None:
        return

    print(f"Inpainting {len(file_ids)} images")

    for idx in tqdm(file_ids):
        in_image_name = os.path.join(input_data_dir, "masked", f"{idx}_stroke_masked.png")
        in_mask_name = os.path.join(input_data_dir, "masks", f"{idx}_stroke_mask.png")
        out_image_name = os.path.join(inpainted_result_dir, f"{idx}.png")

        im_masked = io.imread(in_image_name)
        mask = io.imread(in_mask_name)

        inpainted_image = inpaint_one_image(model, im_masked, mask)
        io.imsave(out_image_name, inpainted_image)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='InpaintImages')
    config = InPaintConfig(args)
    if config.settings is not None:
        inpaint_images(config.settings)
