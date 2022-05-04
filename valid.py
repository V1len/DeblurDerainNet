import torch
from torchvision.transforms import functional as F
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(gopro):
            input_img, label_img, name = data
            input_img = input_img.to(device)

            pred = model(input_img)

            if args.MIMO:
                pred_clip = torch.clamp(pred[2], 0, 1)
            else:
                pred_clip = torch.clamp(pred[0], 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            # print(p_numpy.shape)
            # print(label_numpy.shape)

            p_numpy = (p_numpy * 255).astype(int).transpose(1,2,0)
            label_numpy = (label_numpy * 255).astype(int).transpose(1,2,0)

            # psnr = peak_signal_noise_ratio(p_numpy, label_numpy,data_range=255)
            # ssim = structural_similarity(p_numpy, label_numpy, channel_axis=2)
            psnr = peak_signal_noise_ratio(p_numpy, label_numpy,data_range=255)
            ssim = structural_similarity(p_numpy, label_numpy, channel_axis=2,data_range=255)
            # print(psnr)
            # print(ssim)
            # a==1

            psnr_adder(psnr)
            ssim_adder(ssim)
            print('\r%03d'%idx, end=' ')

            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()

    print('\n')
    model.train()
    return psnr_adder.average(), ssim_adder.average()
