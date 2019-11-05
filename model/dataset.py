import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import dgl_hp as hp
from torch.nn.utils.rnn import pad_sequence


class VocoderDataset(Dataset) :
    def __init__(self, ids, path) :
        self.metadata = ids
        self.sp_cl = path.sp_cl
        self.sp_no = path.sp_no
        self.mag_cl = path.mag_cl


    def __getitem__(self, index) :
        id = self.metadata[index]
        x = np.load(f'{self.sp_no}{id[1]}')
        y = np.load(f'{self.sp_cl}{id[0]}')
        mag = np.load(f'{self.mag_cl}{id[2]}')
        x = torch.from_numpy(x.view(dtype=np.float32).reshape((*x.shape, 2)))
        y = torch.from_numpy(y.view(dtype=np.float32).reshape((*y.shape, 2)))
        mag = torch.from_numpy(mag)
        frame = x.shape[1]
        length = int(id[4])

        return x, y, mag, frame, length

    def __len__(self):
        return len(self.metadata)


def get_vocoder_datasets(path, batch_size):

    with open(path.input_data, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]

    random.seed(1234)
    random.shuffle(metadata)

    test_ids = metadata[-hp.voc_test_samples:]
    train_ids = metadata[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(train_ids, path)
    test_dataset = VocoderDataset(test_ids, path)

    train_set = DataLoader(train_dataset,
                           collate_fn=pad_collate,
                           batch_size=batch_size,
                           num_workers=1,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          collate_fn=pad_collate,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set


def pad_collate(batch):
    B_x = [data[0].permute(1, 0, 2) for data in batch]
    B_y = [data[1].permute(1, 0, 2) for data in batch]
    B_mag = [data[2].permute(1, 0, 2) for data in batch]
    B_f = torch.Tensor([data[3] for data in batch])
    B_len = torch.Tensor([data[4] for data in batch])
    B_x = pad_sequence(B_x, batch_first=True).permute(0, 3, 2, 1)
    B_y = pad_sequence(B_y, batch_first=True).permute(0, 3, 2, 1)
    B_mag = pad_sequence(B_mag, batch_first=True).permute(0, 3, 2, 1)

    return B_x, B_y, B_mag, B_f, B_len

