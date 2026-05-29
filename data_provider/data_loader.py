import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class Dataset_tsicl(Dataset):
    def __init__(self, args, root_path, data_path, size):
        # size = [seq_len, label_len, pred_len]
        self.args = args

        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # self.scaler = StandardScaler()
        data = torch.load(os.path.join(self.root_path, self.data_path), map_location=torch.device('cpu')) # [n_tables, n_samples, sample]
        # print('samples shape:', samples.shape)

        self.n_tables = data.shape[0]
        self.data = data.numpy()

    def __getitem__(self, index):
        seq_x = self.data[index, :, :self.seq_len]
        seq_y = self.data[index, :, self.seq_len - self.label_len:]
        return seq_x, seq_y

    def __len__(self):
        return self.n_tables

    def inverse_transform(self, data):
        # return self.scaler.inverse_transform(data)
        return data


# YGTrip, copy from ygtrip predictor but modified
class Dataset_YGTrip(Dataset):
    def __init__(self, root_path, data_path, line_index, city_index, flag='train',
                 patch_size=120, chunk_size=720, train_n_day=42, test_n_day=7, day_mode='all',
                 scale=True):
        # note, ctx_patch_num and pred_patch_num are not fully implemented and currently fixed to 1
        # if you want to implement these 2 parameters, review all code
        print('initialize ygtrip dataset...')
        self.patch_size = patch_size
        self.chunk_size = chunk_size
        self.valid_chunk_size = chunk_size - patch_size * 2 + 1
        print(f'ygtrip dataset, chunk size: {self.chunk_size}, valid chunk size: {self.valid_chunk_size}')

        self.day_mode = day_mode
        print('day_mode:', day_mode)

        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        cols = list(df_raw.columns)
        data_len = len(df_raw)

        df_extra_info = df_raw[['weekday', 'workday', 'time_index']]
        df_extra_info['line_index'] = [line_index] * data_len
        df_extra_info['city_index'] = [city_index] * data_len
        extra_info_len = len(df_extra_info.columns)

        df_raw = df_raw[df_raw.columns[3:]]
        self.n_stops = len(cols) - 3

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        num_train = chunk_size * train_n_day
        num_test = chunk_size * test_n_day
        num_vali = data_len - num_train - num_test
        print(f'num_train: {num_train}, num_test: {num_test}, num_vali: {num_vali}')

        df_train = df_raw[:num_train]
        df_train_extra_info = df_extra_info[:num_train]

        if self.set_type == 0:
            # train
            df = df_raw[:num_train].reset_index(drop=True)
            df_extra_info = df_extra_info[:num_train].reset_index(drop=True)
            self.chunk_num = num_train // chunk_size
        elif self.set_type == 1:
            # vali, reset index, otherwise tensor concat goes wrong
            df = df_raw[num_train:num_train + num_vali].reset_index(drop=True)
            df_extra_info = df_extra_info[num_train:num_train + num_vali].reset_index(drop=True)
            self.chunk_num = num_vali // chunk_size
        else:
            # test, reset index, otherwise tensor concat goes wrong
            df = df_raw[num_train+num_vali:].reset_index(drop=True)
            df_extra_info = df_extra_info[num_train+num_vali:].reset_index(drop=True)
            self.chunk_num = num_test // chunk_size

        self.scale = scale
        print('dataloader scale', self.scale)
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_train)
            df_train_scaled = self.scaler.transform(df_train)
            df_scaled = self.scaler.transform(df)
            df_train = pd.DataFrame(df_train_scaled, columns=df_raw.columns)
            df = pd.DataFrame(df_scaled, columns=df_raw.columns)

        # use numpy array instead of dateframe
        # [time, stop] -> 3d: [day, time, stop] -> [day, stop, time] -> [stop, day, time]
        self.df_train = df_train.values.reshape(-1, self.chunk_size, self.n_stops).transpose(0, 2, 1).transpose(1, 0, 2)
        self.df = df.values.reshape(-1, self.chunk_size, self.n_stops).transpose(0, 2, 1).transpose(1, 0, 2)

        # [time, extra_info] -> 3d: [day, time, extra_info]
        self.df_train_extra_info = df_train_extra_info.values.reshape(-1, self.chunk_size, extra_info_len)
        self.df_extra_info = df_extra_info.values.reshape(-1, self.chunk_size, extra_info_len)


    def __getitem__(self, idx):
        # day index
        chunk_index = int(idx / (self.valid_chunk_size))
        # time index range
        chunk_offset = idx % (self.valid_chunk_size)
        chunk_end = chunk_offset + self.patch_size

        # [day, time, extra_info], only care about weekday and workday
        extra_info = self.df_extra_info[chunk_index, 0, :]

        if self.day_mode == 'weekday':
            weekday = extra_info[0]
            train_extra_info_weekday = self.df_train_extra_info[:, 0, 0]
            train_day_indices = np.where(train_extra_info_weekday == weekday)[0]
            # [stop, day, time]
            seq_train_x = self.df_train[:, train_day_indices, chunk_offset:chunk_end]
            seq_train_y = self.df_train[:, train_day_indices, chunk_end:chunk_end + self.patch_size]
        elif self.day_mode == 'workday':
            workday = extra_info[1]
            train_extra_info_workday = self.df_train_extra_info[:, 0, 1]
            train_day_indices = np.where(train_extra_info_workday == workday)[0]
            # [stop, day, time]
            seq_train_x = self.df_train[:, train_day_indices, chunk_offset:chunk_end]
            seq_train_y = self.df_train[:, train_day_indices, chunk_end:chunk_end + self.patch_size]
        else:
            # all days
            # [stop, day, time]
            seq_train_x = self.df_train[:, :, chunk_offset:chunk_end]
            seq_train_y = self.df_train[:, :, chunk_end:chunk_end + self.patch_size]

        seq_x = self.df[:, chunk_index:chunk_index+1, chunk_offset:chunk_end]
        seq_y = self.df[:, chunk_index:chunk_index+1, chunk_end:chunk_end + self.patch_size]

        # [stop, train day + 1, time]
        seq_x = np.concatenate((seq_train_x, seq_x), axis=1)
        seq_y = np.concatenate((seq_train_y, seq_y), axis=1)

        return seq_x, seq_y

    def __len__(self):
        return self.chunk_num * self.valid_chunk_size

    # def inverse_transform(self, data):
    #     if self.scale:
    #         return self.scaler.inverse_transform(data)
    #     else:
    #         return data

    def get_scaler(self):
        return self.scaler

