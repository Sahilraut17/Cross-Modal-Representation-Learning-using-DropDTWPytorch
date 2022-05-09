import pickle
import glob
import pandas as pd
import os

PATH = 'dataset'
os.chdir('/common/home/dm1487/Spring22/dropdtw')
if not os.path.exists(PATH):
    os.mkdir(PATH)

def main():
    opj = lambda x, y: os.path.join(x, y)
    image_features = glob.glob('raw_frames/raw_videos/*/*/*.pth')
    text_features = glob.glob('raw_text/training/*/*.pth')
    image_keys, text_keys = [i.split('/')[-1] for i in image_features], [i.split('/')[-1] for i in text_features]
    image_df = pd.DataFrame({'image_features': image_features, 'image_keys': image_keys}).set_index('image_keys')
    text_df = pd.DataFrame({'text_features': text_features, 'text_keys': text_keys}).set_index('text_keys')
    all_data = image_df.join(text_df)
    dataset = [(img, txt) for img, txt in zip(all_data['image_features'].tolist(), all_data['text_features'].tolist())]

    train_ds = []
    valid_ds = []
    for img, txt in dataset:
        if 'training' in img:
            train_ds.append((img, txt))
        if 'validation' in img:
            valid_ds.append((img, txt))

    with open(f'{PATH}/training_data.pkl', 'wb') as f:
        pickle.dump(train_ds, f)

    with open(f'{PATH}/validation_data.pkl', 'wb') as f:
        pickle.dump(valid_ds, f)

if __name__ == '__main__':
    main()