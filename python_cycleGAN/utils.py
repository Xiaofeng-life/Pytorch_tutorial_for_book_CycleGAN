import os
import numpy as np
from PIL import Image


def save_image(image_tensor, out_name, last_acti="tanh"):
    """
    save a single image

    :param image_tensor: torch tensor with size=(3, h, w)
    :param out_name: path+name+".jpg"
    :param last_acti: active function, tanh or sigmoid
    :return: None
    """
    if len(image_tensor.size()) == 3:
        image_numpy = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
        # image_numpy = image_tensor.cpu().detach().numpy()
        if last_acti == "tanh":
            image_numpy = (((image_numpy + 1) / 2) * 255).astype(np.uint8)
        elif last_acti == "sigmod":
            image_numpy = (image_numpy * 255).astype(np.uint8)
        image = Image.fromarray(image_numpy)
        image.save(out_name)
    else:
        raise ValueError("input tensor not with size (3, h, w)")
    return None


def check_mk_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_sub_dirs(base_dir, sub_dirs):
    check_mk_dir(base_dir)
    for sub_dir in sub_dirs:
        check_mk_dir(os.path.join(base_dir, sub_dir))


def make_project_dir(train_dir, val_dir):
    make_sub_dirs(train_dir, ["train_images", "pth", "loss"])
    make_sub_dirs(val_dir, ["val_images"])


if __name__ == "__main__":
    pass
