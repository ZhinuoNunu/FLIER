from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

from timm.data import create_loader
import torch
import torch.utils.data
import torchvision.datasets as datasets
from pathlib import Path

from .labeled_memcached_dataset import *

def build_imagenet_dataset(args, is_train, is_sd, is_fewshot, transform, datasetname):
    if datasetname == 'oxford_pets':
        if args.only_sd_img:
            # 原CLIP模型下的全部sd生成图像的train set
            if is_train:
                dataset = OxfordpetsDataset(args.data_path, 
                    '/home/prp2/zzn/data/oxford_pets/oxfordpets_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = OxfordpetsDataset(args.data_path, 
                    '/home/prp2/zzn/data/oxford_pets/oxfordpets_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            # 加入sd sampler的全部sd生成图像的train set
            if is_train and is_sd and not is_fewshot:
                dataset = OxfordpetsDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/oxford_pets/oxfordpets_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            # 加入sd sampler的few shot实验的sd生成的图像的train set
            elif is_train and is_sd and is_fewshot:
                dataset = OxfordpetsDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/oxford_pets/oxfordpets_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            # 加入sd sampler的全部原数据集的train set
            elif is_train and not is_sd and not is_fewshot:
                dataset = OxfordpetsDataset(args.data_path, 
                    '/home/prp2/zzn/data/oxford_pets/oxfordpets_train.pkl', only_sd_img=False, transform=transform, args=args)
            # 加入sd sampler的few shot实验的原数据集的train set
            elif is_train and not is_sd and is_fewshot:
                dataset = OxfordpetsDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/oxford_pets/oxfordpets_train.pkl', only_sd_img=False, transform=transform, args=args)
            # 验证集
            else:
                # /home/prp2/zzn/data/oxford_pets/oxfordpets_valid.pkl
                dataset = OxfordpetsDataset(args.data_path, 
                    '/home/prp2/zzn/data/oxford_pets/oxfordpets_test.pkl', only_sd_img=False, transform=transform, args=args)        
    elif datasetname == 'caltech':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = CaltechDataset(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = CaltechDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = CaltechDataset(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                dataset = CaltechDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                dataset = CaltechDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                dataset = CaltechDataset(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                dataset = CaltechDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                dataset = CaltechDataset(args.data_path, 
                    '/home/prp2/zzn/data/caltech-101-0126/caltech101_test.pkl', only_sd_img=False, transform=transform, args=args)    
    elif datasetname == 'imagenet':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = ImagenetDataset(args.data_path, 
                    '/home/prp2/zzn/data/imagenet/images/imagenet_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = ImagenetDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/imagenet/images/imagenet_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = ImagenetDataset(args.data_path, 
                    '/home/prp2/zzn/data/imagenet/images/imagenet_valid.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                dataset = ImagenetDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/imagenet/images/imagenet_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                dataset = ImagenetDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/imagenet/images/imagenet_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                dataset = ImagenetDataset(args.data_path, 
                    '/home/prp2/zzn/data/imagenet/images/imagenet_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                dataset = ImagenetDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/imagenet/images/imagenet_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                if args.is_v2:
                    dataset = ImagenetDataset(args.data_path, 
                        '/home/prp2/zzn/data/imagenet/images/imagenet_v2_test.pkl', only_sd_img=False, transform=transform, args=args)
                elif args.is_sketch:
                    dataset = ImagenetDataset(args.data_path, 
                        '/home/prp2/zzn/data/imagenet/images/imagenet_sketch_test.pkl', only_sd_img=False, transform=transform, args=args)
                else:
                    dataset = ImagenetDataset(args.data_path, 
                        '/home/prp2/zzn/data/imagenet/images/imagenet_valid.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'sun397':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                print('MyDatasetSD')
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                print('MyDatasetFSSD')
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                print('MyDatasetFS')
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/sun397/sun397_test.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'food101':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                print('MyDatasetSD')
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                print('MyDatasetFSSD')
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                print('MyDatasetFS')
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/food101/food-101/food101_test.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'dtd':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                print('MyDatasetSD')
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                print('MyDatasetFSSD')
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                print('MyDatasetFS')
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/dtd/dtd/dtd_test.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'flower102':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                print('MyDatasetSD')
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                print('MyDatasetFSSD')
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                print('MyDatasetFS')
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                print('MyDataset')
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/flower102/oxfordflower_test.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'eurosat':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/eurosat/eurosat_test.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'ucf':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/ucf101/ucf101_test.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'stanfordcars':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/stanford_cars/stanfordcars_test.pkl', only_sd_img=False, transform=transform, args=args)
    elif datasetname == 'fgvc':
        if args.only_sd_img:
            if is_train and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_train.pkl', only_sd_img=True, transform=transform, args=args)
            elif is_train and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_train.pkl', only_sd_img=True, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_test.pkl', only_sd_img=False, transform=transform, args=args)
        else:
            if is_train and is_sd and not is_fewshot:
                dataset = MyDatasetSD(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and is_sd and is_fewshot:
                dataset = MyDatasetFSSD(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_train_sd.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and not is_fewshot:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_train.pkl', only_sd_img=False, transform=transform, args=args)
            elif is_train and not is_sd and is_fewshot:
                dataset = MyDatasetFS(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_train.pkl', only_sd_img=False, transform=transform, args=args)
            else:
                dataset = MyDataset(args.data_path, 
                    '/home/prp2/zzn/data/fgvcaircraft/fgvc-aircraft-2013b/data/fgvc_test.pkl', only_sd_img=False, transform=transform, args=args)
    else:
        print("not known dataset")
    return dataset


