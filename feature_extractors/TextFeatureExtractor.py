import cv2
import math
import time
import numpy as np
import os
import glob
import clip
import torch
import pickle

from tqdm import tqdm 
from PIL import Image

opj = lambda x, y: os.path.join(x, y)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("/common/home/dm1487/loaded_models/clip/ViT-B-32.pt", device=device)
os.chdir('/common/home/dm1487/Spring22/dropdtw')
def extract_and_save(steps):
    enc_text = clip.tokenize(steps['sentences']).to(device)
    with torch.no_grad():
        text_features = model.encode_text(enc_text)
    file_dir = opj("raw_text", opj(steps['subset'], steps['recipe_type']))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_path = opj(file_dir, steps['name'] + '.pth')
    torch.save(text_features.cpu(), file_path)

if __name__ == '__main__':
    data_path = 'data/step_texts.pkl'
    data = []
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    for i in tqdm(data):
        extract_and_save(i)
    

