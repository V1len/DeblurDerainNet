import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import pynvml
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


class Core(nn.Module):
    def __init__(self, in_channel, core_size=9):
        super(Core, self).__init__()
        out_channel = 3 * (core_size ** 2)
        self.layers = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            # BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )

    def forward(self, x):
        return self.layers(x)


class Branch(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Branch, self).__init__()
        self.layers = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            # BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )

    def forward(self, x):
        return self.layers(x)


def build_net(args):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg
    
    model_name = args.model_name
    core_size = args.core_size

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    elif model_name == "MIMO-UNet":
        return MIMOUNet()
    
    elif model_name == "DeblurDerainNet":
        return DeblurDerainNet(args=args)


    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')


class DeblurDerainNet(nn.Module):
    def __init__(self, args, num_res = 20):
        super(DeblurDerainNet, self).__init__()
        base_channel = 32
        self.core_size = args.core_size
        self.MIMO = args.MIMO
        self.output_setting = args.output_setting
        self.AddTransformer = args.AddTransformer

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

                    

        if self.MIMO:
            self.FAM1 = FAM(base_channel * 4)
            self.SCM1 = SCM(base_channel * 4)
            self.FAM2 = FAM(base_channel * 2)
            self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

        if "kernel" in self.output_setting:
            if self.MIMO:
                self.Cores = nn.ModuleList(
                    [
                        Core(in_channel=base_channel * 4, core_size=self.core_size),
                        Core(in_channel=base_channel * 2, core_size=self.core_size),
                        Core(in_channel=base_channel * 1, core_size=self.core_size)
                    ]
                )
            else:
                self.Cores = nn.ModuleList(
                    [
                        Core(in_channel=base_channel, core_size=self.core_size),
                    ]
                )

        if "residual" in self.output_setting or "origin" in self.output_setting:
            if self.MIMO:
                self.Branches = nn.ModuleList(
                    [
                        Branch(in_channel=base_channel * 4, out_channel=base_channel * 4),
                        Branch(in_channel=base_channel * 2, out_channel=base_channel * 2),
                        Branch(in_channel=base_channel * 1, out_channel=base_channel * 1)
                    ]
                )
                self.ResidualParts = nn.ModuleList(
                    [
                        BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                        BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
                        BasicConv(base_channel * 1, 3, kernel_size=3, relu=False, stride=1)
                    ]
                )
                
            else:
                self.Branches = nn.ModuleList(
                    [
                        Branch(in_channel=base_channel, out_channel=base_channel)
                    ]
                )
                self.ResidualParts = nn.ModuleList(
                    [
                        BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
                    ]
                )
                

        if self.output_setting == "kernel_residual":
            if self.MIMO:
                self.WeightParts = nn.ModuleList(
                    [
                        BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
                        BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
                        BasicConv(base_channel * 1, 1, kernel_size=3, relu=False, stride=1)
                    ]
                )
            else:
                self.WeightParts = nn.ModuleList(
                    [
                        BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
                    ]
                )
        if self.AddTransformer:
            self.TransformerAFF = TransformerAFF(patch_size=args.patch_size, depth=args.depth, heads=args.heads)

    def pred_by_core(self, x, core, core_size):
        batch_size, N, height, width = x.size()
        core = core.view(batch_size, N, -1, height, width)
        img_stack = []
        padding_num = core_size // 2
        x_pad = F.pad(x, [padding_num, padding_num, padding_num, padding_num])
        for i in range(0, core_size):
            for j in range(0, core_size):
                img_stack.append(x_pad[..., i:i + height, j:j + width])
        img_stack = torch.stack(img_stack, dim=2)
        img_pred = torch.sum(core.mul(img_stack), dim=2, keepdim=False)
        return img_pred

    def forward(self, x):
        if self.MIMO:
            x_2 = F.interpolate(x, scale_factor=0.5)
            x_4 = F.interpolate(x_2, scale_factor=0.5)
            z2 = self.SCM2(x_2)
            z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        if self.MIMO:
            z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        if self.MIMO:
            z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        # res2 = self.drop2(res2)
        # res1 = self.drop1(res1)

        # Transformer
        if self.AddTransformer:
            z21 = F.interpolate(res2, scale_factor=2)
            z41 = F.interpolate(z, scale_factor=4)
            fusion_feature = torch.cat([res1,z21,z41],dim=1)
            res1_part, res2_part, res4_part = self.TransformerAFF(fusion_feature)

            res1 = res1 + res1_part
            res2 = F.interpolate(res2_part, scale_factor=0.5) + res2
            z = F.interpolate(res4_part, scale_factor=0.25) + z



        z = self.Decoder[0](z)
        if self.MIMO:
            if self.output_setting == "origin":
                z_4 = self.Branches[0](z)
                z_4 = self.ResidualParts[0](z_4)
                outputs.append(z_4)
            elif self.output_setting == "residual":
                z_4 = self.Branches[0](z)
                z_4 = self.ResidualParts[0](z_4)
                outputs.append(z_4 + x_4)
            else:
                core_4 = self.Cores[0](z)
                output_4 = self.pred_by_core(x_4,core_4,self.core_size)
                if self.output_setting == "kernel":
                    outputs.append(output_4)
                elif self.output_setting == "kernel_residual":
                    branch_4 = self.Branches[0](z)
                    residual_4 = self.ResidualParts[0](branch_4)
                    weight_4 = nn.Sigmoid()(self.WeightParts[0](branch_4))
                    final_output_4 = weight_4 * output_4 + (1 - weight_4) * residual_4
                    outputs.append(final_output_4)

        z = self.feat_extract[3](z)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        if self.MIMO:
            if self.output_setting == "origin":
                z_2 = self.Branches[1](z)
                z_2 = self.ResidualParts[1](z_2)
                outputs.append(z_2)
            elif self.output_setting == "residual":
                z_2 = self.Branches[1](z)
                z_2 = self.ResidualParts[1](z_2)
                outputs.append(z_2 + x_2)
            else:
                core_2 = self.Cores[1](z)
                output_2 = self.pred_by_core(x_2,core_2,self.core_size)
                if self.output_setting == "kernel":
                    outputs.append(output_2)
                elif self.output_setting == "kernel_residual":
                    branch_2 = self.Branches[1](z)
                    residual_2 = self.ResidualParts[1](branch_2)
                    weight_2 = nn.Sigmoid()(self.WeightParts[1](branch_2))
                    final_output_2 = weight_2 * output_2 + (1 - weight_2) * residual_2
                    outputs.append(final_output_2)
    
        z = self.feat_extract[4](z)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        if self.output_setting == "origin":
            if self.MIMO:
                z_1 = self.Branches[2](z)
                z_1 = self.ResidualParts[2](z_1)
            else:
                z_1 = self.Branches[0](z)
                z_1 = self.ResidualParts[0](z_1)
            outputs.append(z_1)
        elif self.output_setting == "residual":
            if self.MIMO:
                z_1 = self.Branches[2](z)
                z_1 = self.ResidualParts[2](z_1)
            else:
                z_1 = self.Branches[0](z)
                z_1 = self.ResidualParts[0](z_1)
            outputs.append(z_1 + x)
        else:
            if self.MIMO:
                core_1 = self.Cores[2](z)
            else:
                core_1 = self.Cores[0](z)
            output_1 = self.pred_by_core(x,core_1,self.core_size)
            if self.output_setting == "kernel":
                outputs.append(output_1)
            elif self.output_setting == "kernel_residual":
                if self.MIMO:
                    branch_1 = self.Branches[2](z)
                    residual_1 = self.ResidualParts[2](branch_1)
                    weight_1 = nn.Sigmoid()(self.WeightParts[2](branch_1))
                else:
                    branch_1 = self.Branches[0](z)
                    residual_1 = self.ResidualParts[0](branch_1)
                    weight_1 = nn.Sigmoid()(self.WeightParts[0](branch_1))
                final_output_1 = weight_1 * output_1 + (1 - weight_1) * residual_1
                outputs.append(final_output_1)

        return outputs


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerAFF(nn.Module):
    def __init__(self, dim_head = 8, dropout = 0., emb_dropout = 0.,
                patch_size=8, channels = 224,
                depth=4,heads=4):
        super().__init__()
        self.patch_height, self.patch_width = pair(patch_size)

        mlp_dim = channels * self.patch_height * self.patch_width
        dim = channels * self.patch_height * self.patch_width

        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * self.patch_height * self.patch_width

        # print("num_patches: ",num_patches)
        # print("patch_dim: ",patch_dim)
        # a==1

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.Linear(patch_dim, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # print("pos_setting: ", self.pos_embedding.shape)
        # a==1

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()





    def forward(self, img):
        image_height = img.shape[2]
        image_width = img.shape[3]

        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        embedding_to_features = nn.Sequential(
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = image_height // self.patch_height,
                                                            p1 = self.patch_height, p2 = self.patch_width),
        )
        x = embedding_to_features(x)

        return x[:,:32,:,:], x[:,32:32+64,:,:], x[:,32+64:,:,:]