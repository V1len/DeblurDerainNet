import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import argparse
import time
from PIL import Image as Image
from models.MIMOUNet import build_net
import json
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import cv2


import os
input_root = 'blur_rain'
# input_root = 'blur_rain-UNet_BlurRain_SharpRain'
# input_root = 'blur_rain-UNet_BlurRain_Blur'

def WriteJson(my_json, json_path):
    with open(json_path, 'w')as file_obj:
        json.dump(my_json, file_obj)
        file_obj.close()

def ReadJson(json_path):
    with open(json_path, 'r') as load_f:
        load_json = json.load(load_f)
        load_f.close()
    return load_json

def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = path
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
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
        image = Image.open(os.path.join(self.image_dir, input_root, self.map[self.image_list[idx]]))
        # image = Image.open(os.path.join(self.image_dir, input_root, self.image_list[idx]))
        
        label = Image.open(os.path.join(self.image_dir, 'sharp', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            # name = self.image_list[idx]
            name = self.map[self.image_list[idx]]
            return image, label, name
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
    parser.add_argument('--model_name', default='DeblurDerainNet', 
        choices=['MIMO-UNet', 'MIMO-UNetPlus', 'DeblurDerainNet', 'DeblurDerainNetBasic'], type=str)

    # parser.add_argument('--data_dir', type=str, default='dataset/DVD')
    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO/test')
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    # model setting
    parser.add_argument('--core_size', default=9, type=int)
    parser.add_argument('--AFF_core_size', default=9, type=int)
    parser.add_argument('--MIMO', action='store_true')
    parser.add_argument('--AFF_type', default="none", choices=["none", "concat", "kernel", "kernel_residual",
                                                                "single_kernel","single_kernel_residual"])
    parser.add_argument('--output_setting', default="origin", choices=["origin", "kernel", "kernel_residual"])
    parser.add_argument('--gpu_id', required=True)


    # Test
    # parser.add_argument('--test_model', type=str, default='MIMO-UNetPlus.pkl')
    # parser.add_argument('--test_model', type=str, default='results/MIMO-UNetPlus/weights_300_sharpRain_derain/model_300.pkl')

    # parser.add_argument('--test_model', type=str, default='results/MIMO-UNetPlus-AddKernel-Residual/weights_300_coresize_9/Best.pkl')
    # parser.add_argument('--test_model', type=str, default='results/MIMO-UNetPlus/weights_300/Best.pkl')

    # parser.add_argument('--test_model', type=str, default='results/UNet-AddKernel-Residual/weights_300_coresize_9/Best.pkl')
    # parser.add_argument('--test_model', type=str, default='results/UNet/weights_300_onestep/Best.pkl')

    args = parser.parse_args()

    modules_name = ""
    if args.model_name == "DeblurDerainNetBasic":
        modules_name += "Basic-"
    if args.MIMO:
        modules_name += "MIMO-"
    modules_name += "UNet"
    if args.AFF_type == "concat":
        modules_name += "-AFFConcat"
    elif args.AFF_type == "kernel":
        modules_name += "-AFFKernel"
    # elif args.AFF_type == "kernel_group":
    #     modules_name += "-AFFKernelGroup"
    elif args.AFF_type == "kernel_residual":
        modules_name += "-AFFKernelResidual"
    elif args.AFF_type == "single_kernel":
        modules_name += "-AFFSingleKernel"
    elif args.AFF_type == "single_kernel_residual":
        modules_name += "-AFFSingleKernelResidual"

    if args.output_setting == "kernel":
        modules_name += "-Kernel"
    elif args.output_setting == "kernel_residual":
        modules_name += "-KernelResidual"
    elif args.output_setting == "kernel_residual":
        modules_name += "-KernelResidual"

    
    # modules_name += "_BlurRain_Blur"
    # modules_name += "_BlurRain_SharpRain"
    # modules_name += "_SharpRain_Sharp"
    # modules_name += "_Blur_Sharp"
    
    args.modules_name = modules_name    
    args.model_save_dir = os.path.join('results_model/', args.modules_name)
    args.test_model = "{}/Best.pkl".format(args.model_save_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)

    return args

if __name__ == "__main__":
    args = get_args()
    print(args.modules_name)
    model = build_net(args)
    model.cuda()
    my_json = {}
    model_path = args.test_model
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()

    output_root = os.path.join(args.data_dir, '{}-{}'.format(input_root,args.modules_name))
    # output_root = output_root + "_modify"

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    with torch.no_grad():
        psnr_adder = Adder()

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, _ = data
            input_img = input_img.to(device)
            tm = time.time()
            _ = model(input_img)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        count = 0
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            tm = time.time()

            pred = model(input_img)

            if args.MIMO:
                pred_clip = torch.clamp(pred[2], 0, 1)
            else:
                pred_clip = torch.clamp(pred[0], 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            p_numpy = (p_numpy * 255).astype(np.uint8).transpose(1,2,0)

            p_cv2 = cv2.cvtColor(p_numpy,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_root,name[0]), p_cv2)


            # pred_numpy = pred_clip.squeeze(0).cpu()
            
            # vutils.save_image(pred_numpy,os.path.join(output_root,name[0]))


            # if args.save_image:
            #     save_name = os.path.join(args.result_dir, name[0])
            #     pred_clip += 0.5 / 255
            #     pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
            #     print(save_name)
            #     pred.save(save_name)

            # psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            # psnr = peak_signal_noise_ratio(input_numpy, label_numpy, data_range=1)
            # psnr_adder(psnr)
            count += 1
            if (count+1)%100==0:
                print("{}/{}".format(count+1,len(dataloader)))
            # print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        # print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        # print("Average time: %f" % adder.average())
        # # my_json[index] = psnr_adder.average()
        # WriteJson([psnr_adder.average()],"results/eval_compare/before_deblur_{}.json".format(op))
