import os
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net
from train import _train
from eval import _eval

import os

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results_model/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results_model/' + args.modules_name + '/'):
        os.makedirs('results_model/' + args.modules_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)


    model = build_net(args)
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--model_name', default='DeblurDerainNet', 
        choices=['MIMO-UNet', 'MIMO-UNetPlus', 'DeblurDerainNet'], type=str)

    # model setting
    parser.add_argument('--core_size', default=9, type=int)
    parser.add_argument('--MIMO', action='store_true')
    parser.add_argument('--output_setting', default="origin", choices=["origin", "residual", "kernel", "kernel_residual"])
    parser.add_argument('--AddTransformer', action='store_true')


    parser.add_argument('--data_dir', type=str, default='dataset/GOPRO')

    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    parser.add_argument('--gpu_id', required=True)

    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--heads', default=4, type=int)



    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 50 for x in range(300//50)])

    # Test
    parser.add_argument('--test_model', type=str, default='MIMO-UNetPlus.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    
    modules_name = ""
    if args.MIMO:
        modules_name += "MIMO-"
    modules_name += "UNet"

        
    if args.output_setting == "origin":
        modules_name += "-Origin"
    elif args.output_setting == "residual":
        modules_name += "-Residual"    
    elif args.output_setting == "kernel":
        modules_name += "-Kernel"
    elif args.output_setting == "kernel_residual":
        modules_name += "-KernelResidual"

    if args.AddTransformer:
        modules_name += "-Transformer"

    modules_name += "-patchsize{}".format(args.patch_size)
    modules_name += "-depth{}".format(args.depth)
    modules_name += "-heads{}".format(args.heads)

        
    args.modules_name = modules_name 
   


    args.model_save_dir = os.path.join('results_model/', args.modules_name)
    # args.model_save_dir = os.path.join('results_model/', args.modules_name + "_Blur_Sharp")
    # args.model_save_dir = os.path.join('results_model/', args.model_name + "_BlurRain_SharpRain")

    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)

    main(args)
