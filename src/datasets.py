import sqlite3
import random
import json
import h5pickle as h5py
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np


data_path = '/home/hzm/MFC-LLM/data/PHM2012_data.hdf5'
meta_path = '/home/hzm/MFC-LLM/data/PHM2012_data.sqlite'
cache_path = '/home/hzm/MFC-LLM/data/PHM2012_data.json'
corpus_path = '/home/hzm/MFC-LLM/data/PHM2012_data_corpus.json'
train_id_path = '/home/hzm/MFC-LLM/data/PHM2012_data_train_ids.json'
test_id_path = '/home/hzm/MFC-LLM/data/PHM2012_data_test_ids.json'

def get_ref_ids():
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    cursor.execute('SELECT condition_id, file_id FROM file_info WHERE label = 0')
    ref_data = cursor.fetchall()
    conn.close()
    ref_ids = {}
    for condition_id, file_id in ref_data:
        if condition_id not in ref_ids:
            ref_ids[condition_id] = []
        ref_ids[condition_id].append(file_id)
    return ref_ids

def create_cache_dataset():
    ref_ids = get_ref_ids()
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    cursor.execute('SELECT condition_id, file_id, label, rul FROM file_info')
    all_data = cursor.fetchall()
    conn.close()

    label_groups = {}
    for condition_id, file_id, label, rul in all_data:
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append((condition_id, file_id, label, float(rul)))

    dataset_info = {subset: [] for subset in ['train', 'val', 'test']}
    val_id_records = []   
    test_id_records = []  

    for label, samples in label_groups.items():
        random.shuffle(samples)
        n_total = len(samples)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)
        n_test = n_total - n_train - n_val
        subsets = (
            ('train', samples[:n_train]),
            ('val', samples[n_train:n_train+n_val]),
            ('test', samples[n_train+n_val:])
        )
        for subset_name, subset_samples in subsets:
            for condition_id, file_id, label, rul in subset_samples:
                if condition_id in ref_ids:
                    c_ids = ref_ids[condition_id]
                    r_ids = [random.choice(c_ids) for _ in range(3)]
                    for r_id in r_ids:
                        record = [file_id, r_id, label, rul]
                        dataset_info[subset_name].append([file_id, r_id, label, rul])
                        if subset_name == 'train':
                            val_id_records.append({
                                "file_id": file_id,
                                "ref_id": r_id,
                                "label": label,
                                "rul": rul,
                                "condition_id": condition_id
                            })
                        if subset_name == 'test' or subset_name == 'val':
                            test_id_records.append({
                                "file_id": file_id,
                                "ref_id": r_id,
                                "label": label,
                                "rul": rul,
                                "condition_id": condition_id
                            }) 

    with open(cache_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    with open(train_id_path, 'w') as f:
        json.dump(train_id_records, f, indent=2)   

    with open(test_id_path, 'w') as f:
        json.dump(test_id_records, f, indent=2)

def load_cache_dataset():
    if not os.path.exists(cache_path):
        create_cache_dataset()
    with open(cache_path, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

class VibDataset(Dataset):              
    def __init__(self, subset_info):
        self.subset_info = subset_info
        self.data = h5py.File(data_path, 'r')['vibration']

    def __len__(self):
        return len(self.subset_info)

    def __getitem__(self, idx):
        file_id, ref_id, label, rul = self.subset_info[idx]
        data = np.array([self.data[str(file_id)],self.data[str(ref_id)]], dtype=np.float32)
        label = int(label)
        rul = np.float32(rul)
        return data, label, rul

def get_datasets():
    dataset_info = load_cache_dataset()
    train_dataset = VibDataset(dataset_info['train'])
    val_dataset = VibDataset(dataset_info['val'])
    test_dataset = VibDataset(dataset_info['test'])
    return train_dataset, val_dataset, test_dataset

def get_loaders(batch_size, num_workers):
    train_set, val_set, test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

class CorpusDataset:
    def __init__(self):
        self.vib_data = h5py.File(data_path, 'r')['vibration']
        self.corpus = json.load(open(corpus_path, 'r'))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        corpus_data = self.corpus[idx]
        sample_id = corpus_data['id']
        instruction = corpus_data['instruction']
        response = corpus_data['response']
        ref_id = corpus_data['ref_id']
        vib_id = corpus_data['vib_id']

        # 检查 HDF5 是否存在 key
        if str(ref_id) not in self.vib_data:
            raise KeyError(f"ref_id {ref_id} not found in HDF5")
        if str(vib_id) not in self.vib_data:
            raise KeyError(f"vib_id {vib_id} not found in HDF5")

        ref_data = self.vib_data[str(ref_id)]
        vib_data = self.vib_data[str(vib_id)]
        vib = np.array([vib_data, ref_data], dtype=np.float32)
        label_id = corpus_data['label_id']
        rul = np.float32(corpus_data.get('rul', -1))
        return sample_id, label_id, vib, instruction, response, rul

if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_loaders(batch_size=16, num_workers=0)
    print(f'训练样本数: {len(train_loader.dataset)}')
    for data, label, rul in train_loader:
        print('data:', data.shape)     
        print('label:', label.shape)    
        print('rul:', rul.shape)       
        break
