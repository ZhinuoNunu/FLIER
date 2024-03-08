import os
import io
import json
import random

import numpy as np
import pandas as pd

from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        cache (str): class name.
    """

    def __init__(self, impath='', label=0 , cache=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)

        self._impath = impath
        self._label = label
        self._cache = cache

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def cache(self):
        return self._cache

def split_dataset_by_label(data_source):
    output = defaultdict(list)

    for item in data_source:
        output[item.label].append(item)

    return output

def generate_fewshot_dataset(*data_sources, num_shots=-1, repeat=True):
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = split_dataset_by_label(data_source)
            dataset_shot = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)  # num_shots
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset_shot.extend(sampled_items)

            output.append(dataset_shot)

        if len(output) == 1:
            return output[0]

        return output

class OxfordpetsDataset(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open('./data/oxfordpets.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])

        self.A_paths = train['image'].to_list()
        self.A_labels = train['label'].to_list()

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        return A, A_label
    
class OxfordpetsDatasetSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open('./data/oxfordpets.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        
        self.A_paths = train['image'].to_list()
        self.A_caches = train['cache'].to_list()     
        self.A_labels = train['label'].to_list()

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache

class OxfordpetsDatasetFSSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open('./data/oxfordpets.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        
        # few shot setting
        data_paths = train['image'].to_list()
        data_caches = train['cache'].to_list()     
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                    cache=data_caches[i]
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_caches.append(train_fewshot[i].cache)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache

class OxfordpetsDatasetFS(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open('./data/oxfordpets.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])
        
        # self.A_paths = train['image'].to_list()   
        # self.A_labels = train['label'].to_list()
        
        # few shot setting
        data_paths = train['image'].to_list()    
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]

        return A, A_label

class CaltechDataset(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open('./data/caltech101.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])
            
        self.A_paths = train['image'].to_list()
        train_images_label = train['label'].to_list()
        tmp = []
        for i in range(len(train_images_label)):
            tmp.append(train_images_label[i])
        self.A_labels = tmp

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        return A, A_label
    
class CaltechDatasetSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open('./data/caltech101.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)

        self.A_paths = train['image'].to_list()
        self.A_caches = train['cache'].to_list()     
        self.A_labels = train['label'].to_list()

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache

class CaltechDatasetFSSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open('./data/caltech101.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        
        # few shot setting
        data_paths = train['image'].to_list()
        data_caches = train['cache'].to_list()     
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                    cache=data_caches[i]
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_caches.append(train_fewshot[i].cache)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache

class CaltechDatasetFS(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open('./data/caltech101.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])
        
        # few shot setting
        data_paths = train['image'].to_list()    
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]

        return A, A_label

class ImagenetDataset(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open('./data/imagenet.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])
            
        self.A_paths = train['image'].to_list()
        train_images_label = train['label'].to_list()
        tmp = []
        for i in range(len(train_images_label)):
            tmp.append(train_images_label[i])
        self.A_labels = tmp

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        return A, A_label
    
class ImagenetDatasetSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open('./data/imagenet.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)

        self.A_paths = train['image'].to_list()
        self.A_caches = train['cache'].to_list()     
        self.A_labels = train['label'].to_list()

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache
    
class ImagenetDatasetFSSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open('./data/imagenet.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        
        # few shot setting
        data_paths = train['image'].to_list()
        data_caches = train['cache'].to_list()     
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                    cache=data_caches[i]
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_caches.append(train_fewshot[i].cache)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache

class ImagenetDatasetFS(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open('./data/imagenet.json', 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])
        
        # few shot setting
        data_paths = train['image'].to_list()    
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]

        return A, A_label
    
    
    
    
class MyDataset(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open(args.dict_path, 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])
            
        self.A_paths = train['image'].to_list()
        train_images_label = train['label'].to_list()
        tmp = []
        for i in range(len(train_images_label)):
            tmp.append(train_images_label[i])
        self.A_labels = tmp

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        return A, A_label
    
class MyDatasetSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        
        class_dist = json.load(open(args.dict_path, 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)

        self.A_paths = train['image'].to_list()
        self.A_caches = train['cache'].to_list()     
        self.A_labels = train['label'].to_list()

        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache
    
class MyDatasetFSSD(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open(args.dict_path, 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        
        # few shot setting
        data_paths = train['image'].to_list()
        data_caches = train['cache'].to_list()     
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                    cache=data_caches[i]
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_caches.append(train_fewshot[i].cache)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]
        A_cache = self.A_caches[index % self.A_size]
        return A, A_label, A_cache

class MyDatasetFS(Dataset):
    def __init__(self, root, input_file, phase = 'train', only_sd_img=False, transform=None, args=None):
        self.transform = transform
        self.labels = {}
        self.shot = args.shot
        
        class_dist = json.load(open(args.dict_path, 'r'))
        
        keys = list(class_dist.keys())
        values = []
        for i in range(len(keys)):
            values.append(class_dist[keys[i]])

        for i in range(len(values)):
            self.labels[values[i]] = keys[i]

        train = pd.read_pickle(input_file)
        if only_sd_img:
            sd_file = input_file.replace('.pkl','_sd.pkl')
            train_sd = pd.read_pickle(sd_file)
            train = pd.concat([train, train_sd])
        
        # self.A_paths = train['image'].to_list()   
        # self.A_labels = train['label'].to_list()
        
        # few shot setting
        data_paths = train['image'].to_list()    
        data_labels = train['label'].to_list()
        
        data_sources = []
        for i in range(len(data_paths)):
            item = Datum(
                    impath=data_paths[i],
                    label=data_labels[i],
                )
            data_sources.append(item)
            
        train_fewshot = generate_fewshot_dataset(data_sources, num_shots=self.shot)
        
        self.A_paths = []
        self.A_caches = [] 
        self.A_labels = []
        for i in range(len(train_fewshot)):
            self.A_paths.append(train_fewshot[i].impath)
            self.A_labels.append(train_fewshot[i].label)
        
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        try:
            return self.load_img(index)
        except:
            return self.__getitem__(random.randint(0, self.__len__()-1))

    def load_img(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        A_label = self.A_labels[index % self.A_size]

        return A, A_label