from PIL import Image
from torch.utils.data import Dataset
import os
import random
import numpy as np

class FaceForensics(Dataset):
    def __init__(self, img_txt, compression, transform):
        super(FaceForensics, self).__init__()

        root_dir = ""
        crops_dir = os.path.join(root_dir, 'crops', compression)
        lists_dir = os.path.join(root_dir, 'lists')

        ms = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

        with open(os.path.join(lists_dir, img_txt)) as f:
            nums = [a.strip() for a in f.readlines()]

        imgs = []

        for num in nums:
            i = num.split('_')[0]
            original_video = os.path.join(crops_dir, 'original_sequences', 'youtube', i)
            o_imgs = [os.path.join(original_video, img) for img in os.listdir(original_video)]
            for img in o_imgs:
                imgs.append((img, 1))
            for m in ms:
                manipulated_video = os.path.join(crops_dir, 'manipulated_sequences', m, num)
                m_imgs = [os.path.join(manipulated_video, img) for img in os.listdir(manipulated_video)]
                for img in m_imgs:
                    imgs.append((img, 0))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class FaceShifter(Dataset):
    def __init__(self, img_txt, compression, transform):
        super(FaceShifter, self).__init__()

        root_dir = ""
        crops_dir = os.path.join(root_dir, 'FaceForensics++')
        lists_dir = '/data/users/hanyiwang/data/DeepFake/FaceForensics++/lists'

        ms = ["FaceShifter"]

        with open(os.path.join(lists_dir, img_txt)) as f:
            nums = [a.strip() for a in f.readlines()]

        imgs = []

        for num in nums:
            i = num.split('_')[0]
            original_video = os.path.join(crops_dir, 'original_sequences', 'youtube', compression, 'frames', i)
            o_imgs = [os.path.join(original_video, img) for img in os.listdir(original_video)]
            for img in o_imgs:
                imgs.append((img, 1))
            for m in ms:
                manipulated_video = os.path.join(crops_dir, 'manipulated_sequences', m, compression, 'frames', num)
                m_imgs = [os.path.join(manipulated_video, img) for img in os.listdir(manipulated_video)]
                for img in m_imgs:
                    imgs.append((img, 0))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class DeepFakeDetection(Dataset):
    def __init__(self, compression, transform):
        super(DeepFakeDetection, self).__init__()

        root_dir = ""
        crops_dir = os.path.join(root_dir, 'FaceForensics++')

        ms = ["DeepFakeDetection"]

        with open(os.path.join(lists_dir, img_txt)) as f:
            nums = [a.strip() for a in f.readlines()]

        imgs = []

        for num in nums:
            i = num.split('_')[0]
            original_video = os.path.join(crops_dir, 'original_sequences', 'youtube', compression, 'frames', i)
            o_imgs = [os.path.join(original_video, img) for img in os.listdir(original_video)]
            for img in o_imgs:
                imgs.append((img, 1))
            for m in ms:
                manipulated_dir = os.path.join(crops_dir, 'manipulated_sequences', m, compression, 'frames')
                for manipulated_video in os.listdir(manipulated_dir):
                    m_imgs = [os.path.join(manipulated_dir, manipulated_video, img) for img in os.listdir(os.path.join(manipulated_dir, manipulated_video))]
                    for img in m_imgs:
                        imgs.append((img, 0))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

class Celeb_DF_v2(Dataset):
    def __init__(self, transform):
        super(Celeb_DF_v2, self).__init__()

        root_dir = ''
        test_txt = os.path.join(root_dir, 'List_of_testing_videos.txt')

        imgs = []

        with open(test_txt, 'r') as file:
            for line in file:
                line = line.strip()
                label, filename = line.split(' ', 1)
                video_name = filename.split('.')[0]
                label = int(label)
                video_path = os.path.join(root_dir, video_name)
                for frame in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame)
                    imgs.append((frame_path, label))

        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class WildDeepFake(Dataset):
    def __init__(self, transform):
        super(WildDeepFake, self).__init__()
        root_dir = ""
        real_dir = os.path.join(root_dir, 'real_test')
        fake_dir = os.path.join(root_dir, 'fake_test')

        imgs = []

        for id in os.listdir(real_dir):
            id_dir = os.path.join(real_dir, id, 'real')
            for num in os.listdir(id_dir):
                real_imgs = [os.path.join(id_dir, num, img) for img in os.listdir(os.path.join(id_dir, num))]
                for img in real_imgs:
                    imgs.append((img, 1))

        for id in os.listdir(fake_dir):
            id_dir = os.path.join(fake_dir, id, 'fake')
            for num in os.listdir(id_dir):
                fake_imgs = [os.path.join(id_dir, num, img) for img in os.listdir(os.path.join(id_dir, num))]
                for img in fake_imgs:
                    imgs.append((img, 0))


        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
