import cv2
import math
import time
import numpy as np
import os
import glob
import clip
import torch

from tqdm import tqdm 
from PIL import Image

opj = lambda x, y: os.path.join(x, y)


device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/common/home/dm1487/loaded_models/clip/ViT-B-32.pt", device=device)

os.chdir('/common/home/dm1487/Spring22/dropdtw')

def extract_and_save(full_path):
    batch_size = 500
    path, dest, vid = '/'.join(full_path.split('/')[:-1]), '/'.join(full_path.split('/')[4:-1]), full_path.split('/')[-1]

    video = cv2.VideoCapture(opj(path, vid))
    start_time = time.time()
    frames = []
    get_frame_at = math.ceil(video.get(cv2.CAP_PROP_FRAME_COUNT)/500)
    start = 0
    features = torch.tensor([])
    extracted_features = torch.tensor([])
    while video.isOpened():
        if features.shape[0] == batch_size:
            with torch.no_grad():
                image_features = model.encode_image(features)
                if extracted_features.shape[0] == 0:
                    extracted_features = image_features.cpu()
                else:
                    extracted_features = torch.vstack([extracted_features, image_features.cpu()])
                    print(extracted_features.size())
            features = torch.tensor([])
        ret, frame = video.read()
        if ret == True:
            if start % get_frame_at == 0:
                image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
                if features.shape[0] == 0:
                    features = image
                else:
                    features = torch.vstack([features, image])
            start += 1
        else:
            if features.shape[0] > 0:
                with torch.no_grad():
                    image_features = model.encode_image(features)
                    if extracted_features.shape[0] == 0:
                        extracted_features = image_features.cpu()
                    else:
                        extracted_features = torch.vstack([extracted_features, image_features.cpu()])
            break
    if not os.path.exists(opj("raw_frames", dest)):
        os.makedirs(opj("raw_frames", dest))

    new_path = opj("raw_frames", dest)
    torch.save(extracted_features, opj(new_path, ''.join((vid.split('.mp4')[0], '.pth'))))

if __name__ == '__main__':
    all_files = glob.glob('/common/users/dm1487/raw_videos/testing/*/*.mp4')

    completed = []
    if os.path.exists('done_test.txt'):
        with open('done_test.txt', 'r') as f:
            completed = f.readlines()

    completed = [i.strip() for i in completed]

    for i in tqdm(all_files):
        if i in completed:
            print('skipping', i)
            continue
        extract_and_save(i)
        with open('done_test.txt', 'a') as f:
            f.write(i + '\n')
