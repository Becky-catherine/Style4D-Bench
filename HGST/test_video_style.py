# -*- coding: utf-8 -*-

import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename, splitext
from torchvision import transforms
from torchvision.utils import save_image
import net
import numpy as np
import cv2
import yaml


# 读取文件函数
def get_files(img_dir):
    files = os.listdir(img_dir)
    paths = []
    for x in files:
        paths.append(os.path.join(img_dir, x))
    return paths


# 加载图片
def load_images(args):
    assert (args.content or args.content_dir)
    assert (args.style or args.style_dir)
    if not args.content:
        content_paths = get_files(args.content_dir)  # 修改为 args.content_dir
    else:
        content_paths = [args.content]
    if not args.style:
        style_paths = get_files(args.style_dir)  # 修改为 args.style_dir
    else:
        style_paths = [args.style]
    return content_paths, style_paths


# 加载模型参数
def load_weights(vgg, decoder, mcc_module):
    vgg.load_state_dict(torch.load(args.vgg_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    mcc_module.load_state_dict(torch.load(args.transform_path))


# 图片预处理
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# 风格图预处理
def style_transform(h, w):
    k = (h, w)
    size = int(np.max(k))

    transform_list = []
    transform_list.append(transforms.Resize(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# 内容图预处理
def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# 风格化函数
def style_transfer(vgg, decoder, sa_module, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)

    style_fs, content_f, style_f = feat_extractor(vgg, content, style)

    Fccc = sa_module(content_f, content_f)

    if interpolation_weights:
        _, C, H, W = Fccc.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = sa_module(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        Fccc = Fccc[0:1]
    else:
        feat = sa_module(content_f, style_f)
    feat = feat * alpha + Fccc * (1 - alpha)
    return decoder(feat), feat


def feat_extractor(vgg, content, style):
    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)

    content3_1 = enc_3(enc_2(enc_1(content)))
    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
    Content5_1 = enc_5(Content4_1)
    Style3_1 = enc_3(enc_2(enc_1(style)))
    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)

    content_f = [content3_1, Content4_1, Content5_1]
    style_f = [Style3_1, Style4_1, Style5_1]

    style_fs = [enc_1(style), enc_2(enc_1(style)), enc_3(enc_2(enc_1(style))), Style4_1, Style5_1]

    return style_fs, content_f, style_f


# 图片处理，调用style_transfer
def image_process(content, style, type="image"):
    if type == "image":
        content_tf1 = content_transform()
        content_frame = content_tf1(content.convert("RGB"))
        h, w, c = np.shape(content_frame)
        style_tf1 = test_transform(512, True)
        style = style_tf1(style.convert("RGB"))

    else:
        content_tf1 = content_transform()
        content_frame = content_tf1(content)

        h, w, c = np.shape(content_frame)
        style_tf1 = test_transform(512, True)
        style = style_tf1(style.convert("RGB"))

    if yaml['preserve_color']:
        style = coral(style, content)

    style = style.to(device).unsqueeze(0)
    content = content_frame.to(device).unsqueeze(0)

    with torch.no_grad():
        output, style_f = style_transfer(vgg, decoder, mcc_module, content, style, alpha)

    return output.squeeze(0).cpu(), style_f


# 图像风格化并保存
def process_image(content_path, style_path, style_folder):
    # 使用style图像的原名称创建文件夹
    style_name = splitext(basename(style_path))[0]
    style_folder_path = os.path.join(style_folder, style_name)
    
    if not os.path.exists(style_folder_path):
        os.makedirs(style_folder_path)

    # 保存风格化后的content图像
    content_name = splitext(basename(content_path))[0]
    image_name = os.path.join(style_folder_path, f'{content_name}.png')

    # 对图像进行风格迁移
    content = Image.open(content_path)
    style = Image.open(style_path)
    output, style_f = image_process(content, style, "image")

    save_image(output, image_name)
    print(f"Saved stylized image: {image_name}")


def test(content_paths, style_paths):
    for style_path in style_paths:
        # 为每个style图像创建一个文件夹，并处理所有content图像
        style_folder = output_path  # 存储所有风格化图像的根目录
        
        for content_path in content_paths:
            process_image(content_path, style_path, style_folder)


def create_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, default="",
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str, default="",
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do style \
                        interpolation or spatial control')
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')
    parser.add_argument('--vgg_path', type=str, default='./checkpoints/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default='./checkpoints/decoder_iter_160000.pth')
    parser.add_argument('--transform_path', type=str, default='./checkpoints/mcc_module_iter_160000.pth')
    parser.add_argument('--yaml_path', type=str, default='./yaml/test.yaml')
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--style_interpolation_weights', type=str, default="")

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    return args


if __name__ == '__main__':
    args = create_args()
    with open(args.yaml_path, 'r') as file:
        yaml = yaml.load(file, Loader=yaml.FullLoader)
    alpha = args.a
    output_path = args.output
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    decoder = net.decoder
    vgg = net.vgg
    vgg = vgg.to(device)

    network = net.Net(vgg, decoder)
    mcc_module = network.mcc_module
    decoder.eval()
    mcc_module.eval()
    vgg.eval()
    load_weights(vgg, decoder, mcc_module)

    norm = nn.Sequential(*list(vgg.children())[:1])
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    norm.to(device)
    enc_1.to(device)
    enc_2.to(device)
    enc_3.to(device)
    enc_4.to(device)
    enc_5.to(device)
    mcc_module.to(device)
    decoder.to(device)
    
    content_paths, style_paths = load_images(args)
    output_path = args.output

    test(content_paths, style_paths)
