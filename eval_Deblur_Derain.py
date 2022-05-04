import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import argparse
import time
from PIL import Image as Image
from models.MIMOUNet import build_net
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import cv2


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


ops = ["-MIMO-UNet-AFFKernelResidual-KernelResidual","-MIMO-UNet-AFFKernelResidual-Kernel",
        "-MIMO-UNet-AFFKernel-KernelResidual","-MIMO-UNet-AFFKernel-Kernel"]

ops = ["-MIMO-UNet-AFFSingleKernelResidual-KernelResidual"]

def WriteJson(my_json, json_path):
    with open(json_path, 'w')as file_obj:
        json.dump(my_json, file_obj)
        file_obj.close()

def ReadJson(json_path):
    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
        load_f.close()
    return load_json

def test_dataloader(path,input_root, batch_size=1, num_workers=0):
    image_dir = path
    dataloader = DataLoader(
        DeblurDataset(image_dir,input_root, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

class DeblurDataset(Dataset):
    def __init__(self, image_dir,input_root, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'sharp/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.map = ReadJson("dataset/rain_map_test.json")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, input_root, self.map[self.image_list[idx]]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def get_args():
    parser = argparse.ArgumentParser()
     # Directories
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO/test')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # Test
    parser.add_argument('--test_model', type=str, default='MIMO-UNetPlus.pkl')
    args = parser.parse_args()


    return args

if __name__ == "__main__":
    args = get_args()
    # model_path = "results/MIMO-UNetPlus/adv_opinsubmotion_finetune_200/model_{}.pkl".format(197)

    for op in ops:
        input_root = "blur_rain{}".format(op)
        print(input_root)
        dataloader = test_dataloader(args.data_dir,input_root, batch_size=1, num_workers=0)
        adder = Adder()

        with torch.no_grad():
            psnr_adder = Adder()
            ssim_adder = Adder()

            # Main Evaluation
            count = 0
            for iter_idx, data in enumerate(dataloader):
                input_img, label_img = data
                input_img = input_img[0].numpy()
                label_img = label_img[0].numpy()

                psnr = peak_signal_noise_ratio(input_img, label_img)
                ssim = structural_similarity(input_img, label_img, channel_axis=2)
                psnr_adder(psnr)
                ssim_adder(ssim)

                count += 1
                # if (count+1)%100==0:
                #     print("{}/{}".format(count+1,len(dataloader)))
                # print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

            print('==========================================================')
            print('The average PSNR is %.2f dB' % (psnr_adder.average()))
            print('The average SSIM is %.2f dB' % (ssim_adder.average()))
            # print("Average time: %f" % adder.average())
            # # my_json[index] = psnr_adder.average()

            # WriteJson([psnr_adder.average()],"results/eval_compare/rainy_blur.json")
            WriteJson({"PSNR":psnr_adder.average(),"SSIM":ssim_adder.average()},"results_model/eval_compare/{}.json".format(input_root))
