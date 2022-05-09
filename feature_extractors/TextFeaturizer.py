from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModel ##
from datasets import Dataset ##
from pathlib import Path
from tqdm import tqdm 

import os
import pandas as pd
import pickle
import torch

MODEL_PATH = '/common/home/dm1487/loaded_models/bert-base-uncased' 


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.dataset[idx]['input_ids']).long(), 'attention_mask': torch.tensor(self.dataset[idx]['attention_mask']).float()}



def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) ##
    model = AutoModel.from_pretrained(MODEL_PATH, output_hidden_states=True)

    def tokenize(batch): ##
        return tokenizer(batch['sentence'], padding=True) ##

    data_path = Path('data')
   
    if not os.path.exists(data_path/'tokenized_text.pkl'):
        df = pd.read_csv(data_path/'youcook_text.csv') ##
        dataset = Dataset.from_pandas(df) ##
        enc_dataset = dataset.map(tokenize, batched=True, batch_size=None) ##
        with open(data_path/'tokenized_text.pkl', 'wb') as f:
            pickle.dump(enc_dataset, f)
    else:
        with open(data_path/'tokenized_text.pkl', 'rb') as f:
            enc_dataset = pickle.load(f)

    
    '''
    enc_dataset.set_format('pandas')
    df = enc_dataset[:]
    '''
    
    batch_size = 512
    dataset_torch = TextDataset(enc_dataset)
    dataloader = torch.utils.data.DataLoader(dataset_torch, batch_size, shuffle=False)
    model = model.to('cuda')
    parallel_model = torch.nn.DataParallel(model)

    features_folder = data_path/'features_text_instructions'
    if not os.path.exists(features_folder):
        os.mkdir(features_folder)

    with torch.no_grad():
        features = torch.tensor([])
        for idx, batch in tqdm(enumerate(dataloader)):
            output =  parallel_model(**batch)
            states = output[2][-2]
            # print(states.size())
            if features.shape[0] == 0:
                features = torch.mean(states, dim=1)
            else:
                features = torch.vstack([features, torch.mean(states, dim=1)])
        torch.save(features, features_folder/'all_text_features.pkl') 

if __name__ == '__main__':

    main()
