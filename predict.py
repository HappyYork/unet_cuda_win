import os
import time
import json

import torch
import numpy as np
from PIL import Image
from unet import *
import argparse


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def parse_args():

    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="..\\data\\", help="VOCdevkit root")           #modified the root path
    parser.add_argument("--num-classes", default=20, type=int)
    args = parser.parse_args()
    return args

def get_para_num(model):
    names = []
    paras = []
    for name,para in model.named_parameters():
        names.append(name)
        paras.append(para.numel())
    return names,paras


def main():

    weights_path ='save_weights\\model_12450.pth'
    img_path = ".\\test2.jpg"
    result_path = ".\\test2.png"

    palette_path = ".\\palette.json"

    base_size,crop_size = 520,256

    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette.append(v)

    # load image
    original_img = Image.open(img_path)
    target_img = Image.open(result_path)

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    args = parse_args()
    num_classes = args.num_classes + 1
    model = create_model(num_classes)

    weights_dict = torch.load(weights_path, map_location='cuda')['model']

    model.load_state_dict(weights_dict)
    model.to(device)

    seg = get_transform(False)
    resized_image = original_img.resize((256,256),Image.LANCZOS)
    resized_target = target_img.resize((256,256),Image.LANCZOS)
    clipped_img,clipped_target = seg(resized_image,resized_target)
    # expand batch dimension
    img = torch.unsqueeze(clipped_img, dim=0)

    model.eval()  # 进入验证模式
    correct = 0
    with torch.no_grad():
        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(np.array(pallette, dtype=np.uint8))
        mask.save("target_test.png")

        result = clipped_target.to("cpu").numpy().astype(np.uint8)

        correct = (prediction == result).sum()

        print("accuracy:{:.2f}%".format((100*correct)/(crop_size**2)))

        result = Image.fromarray(result)
        result.putpalette(np.array(pallette, dtype=np.uint8))
        result.save("target_orgin.png")

        names,total_params = get_para_num(model)
        print(names)
        print(total_params)
        print("params:{}".format(sum(total_params)))
if __name__ == '__main__':
    main()
