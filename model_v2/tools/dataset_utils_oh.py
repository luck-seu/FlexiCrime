# -*- coding: utf-8 -*-
# @File    : dataset_utils.py
# @Author  : juntang
# @Time    : 2023/03/04 15:02
# 因为一下要输入多种类型，需要启用类别的one-hot编码
import pickle

import torch
import random
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from experiment.NYC.exp_args import exp_args


class BaseDataSet(torch.utils.data.Dataset):
    def __init__(self, city_base_info, size, crime_dataset_all, predict_crime_id=-1, need_shuffle=False, self_adjustment=False, p_ratio=0.5):
        super().__init__()
        self.predict_crime_id = predict_crime_id
        self.n_lon_len, self.n_lat_len = city_base_info['n_lon_len'], city_base_info['n_lat_len']
        self.n_POI_cate = city_base_info['n_POI_cate']
        self.n_crime_type = city_base_info['n_crime_type']
        self.n_grid = city_base_info['n_grid']
        self.POI_feature_cate = city_base_info['POI_feature_cate']
        # 时间为x，类型为y，值为crime one-hot 和 grid feature 拼接 -> 训练时crime one-hot已经弃用，先放着以防要用吧
        self.C_time_type = self._update_map(city_base_info['crimeMap_time_crime'])
        self.time_type_mask = city_base_info['time_crime_mask']
        self.time_crime_grid_coor = city_base_info['time_crime_grid_coor']

        self.crime_dataset = crime_dataset_all[0]
        self.query_time_list = crime_dataset_all[1]
        self.prediction_query_time_list = crime_dataset_all[2]

        self.time_list = city_base_info['time_list']
        self._normalization_time()

        self._crime_dataset_len = len(self.crime_dataset)
        self._size = min(self._crime_dataset_len, size)

        self.generate_id = 0
        if self_adjustment:
            self._self_adjust_crime_dataset(p_ratio)
        # if adjustment:
        #     # 对数据进行调整，让正样本（即犯罪事件）得到充分训练->控制训练集中正样本的比例
        #     self._adjust_crime_dataset(p_ratio)
        if need_shuffle:
            random.seed(exp_args.seed)
            random.shuffle(self.crime_dataset)

    def __len__(self):
        return self._size

    def info(self):
        neg_num, pos_num = 0, 0
        for cr in self.crime_dataset:
            if cr[-1] == 0:
                neg_num += 1
            else:
                pos_num += 1
        print("pos_num:", pos_num, "neg_num:", neg_num)

    def _self_adjust_crime_dataset(self, p_ratio):
        # 让犯罪样本得到充足训练
        pos_dataset, neg_dataset = [], []
        for cr in self.crime_dataset:
            if cr[-1] == 0:
                neg_dataset.append(cr)
            else:
                pos_dataset.append(cr)
        # print(len(pos_dataset), len(neg_dataset))
        # 调整crime_dataset, 按照pn_ratio来调整，大于等于pn_ratio。因为正样本的数据有限，不一定能达到
        if len(pos_dataset) / len(neg_dataset) > p_ratio:
            return
        else:
            # pos_num = min(math.ceil(self._size * p_ratio), len(pos_dataset))
            pos_num = math.ceil(self._size * p_ratio)
            neg_num = self._size - pos_num
            if pos_num <= len(pos_dataset):
                now_crime_dataset = pos_dataset[:pos_num] + neg_dataset[:neg_num]
            else:
                now_crime_dataset = neg_dataset[:neg_num]
                while pos_num != 0:
                    add_len = min(pos_num, len(pos_dataset))
                    now_crime_dataset = now_crime_dataset + pos_dataset[:add_len]
                    pos_num -= add_len
            # neg_num = self._size - pos_num
            # now_crime_dataset = pos_dataset[:pos_num] + neg_dataset[:neg_num]
            for i in range(3):
                random.seed(exp_args.seed)
                random.shuffle(now_crime_dataset)
            self.crime_dataset = now_crime_dataset

    def _adjust_crime_dataset(self, p_ratio):
        pos_dataset, neg_dataset = [], []
        for cr in self.crime_dataset:
            if cr[-1] == 0:
                neg_dataset.append(cr)
            else:
                pos_dataset.append(cr)
        # 调整crime_dataset, 按照pn_ratio来调整，确保不大于pn_ratio。因为真样本的数据有限，不一定能达到
        pos_num = min(math.ceil(self._size * p_ratio), len(pos_dataset))
        neg_num = self._size - pos_num
        now_crime_dataset = pos_dataset[:pos_num] + neg_dataset[:neg_num]
        random.shuffle(now_crime_dataset)
        self.crime_dataset = now_crime_dataset

    def _normalization_time(self):
        # normalize time_list and query_time_list
        all_time = []
        all_time.extend(self.time_list.tolist())
        for a_q_t in self.query_time_list:
            for q_t in a_q_t:
                all_time.extend(q_t.tolist())
        for q_t in self.prediction_query_time_list.tolist():
            all_time.extend(q_t)
        all_time.sort()
        all_time = np.array(list(set(all_time)))
        min_max_scaler = MinMaxScaler()

        all_time_len = len(all_time)
        all_time = all_time.reshape((all_time_len, -1))
        min_max_scaler = min_max_scaler.fit(all_time)

        # transfer time list
        time_list_len = len(self.time_list)
        self.time_list = min_max_scaler.transform(self.time_list.reshape(time_list_len, -1)).reshape(time_list_len)

        # transfer query time list
        self.query_time_list = self.query_time_list.astype(np.float64)
        for i in range(len(self.query_time_list)):
            for j in range(len(self.query_time_list[i])):
                now_len = len(self.query_time_list[i, j])
                self.query_time_list[i, j] = min_max_scaler.transform(self.query_time_list[i, j].reshape(now_len, -1)). \
                    reshape(now_len)
        # transfer prediction query time list
        self.prediction_query_time_list = self.prediction_query_time_list.astype(np.float64)
        for i in range(len(self.prediction_query_time_list)):
            now_len = len(self.prediction_query_time_list[i])
            self.prediction_query_time_list[i] = min_max_scaler.transform(self.prediction_query_time_list[i].reshape(now_len, -1)). \
                reshape(now_len)

        # 过小，放缩一下
        self.time_list = self.time_list * 10
        self.query_time_list = self.query_time_list * 10
        self.prediction_query_time_list = self.prediction_query_time_list * 10

    def _update_map(self, n_map):
        # 调整为crime id的one-hot向量和 grid_id 对应feature的拼接
        x, y, z = n_map.shape
        new_map = np.zeros((x, y, self.n_crime_type + self.n_POI_cate))
        for i in range(x):
            for j in range(y):
                crime_id, grid_id = n_map[i, j]
                if grid_id == -1 or crime_id == -1:
                    continue
                crime_one_hot = np.zeros(self.n_crime_type)
                crime_one_hot[crime_id] = 1
                grid_feature = self.POI_feature_cate[grid_id]
                new_crime_feature = np.concatenate((crime_one_hot, grid_feature))
                new_map[i, j] = new_crime_feature
        return new_map

    def _get_generate_id(self):
        now_id = self.generate_id
        self.generate_id = (self.generate_id + 1) % self._size
        return now_id

    def _get_grid_id(self, grid_coor):
        return grid_coor[0] * self.n_lat_len + grid_coor[1]

    def _find_id(self, q_t, k_t):
        # 在k_t中找到第一个大于q_t的位置
        le, ri = 0, len(k_t) - 1
        id = len(k_t)
        while le <= ri:
            mid = (le + ri) // 2
            if k_t[mid] > q_t:
                id = mid
                ri = mid - 1
            else:
                le = mid + 1
        return id

    def _generate_data(self):
        id = self._get_generate_id()
        cd = self.crime_dataset[id]
        in_t_s_id, in_t_e_id, q_t_id, query_grid, label = cd[0], cd[1], cd[2], cd[3], cd[4]
        query_time = self.query_time_list[q_t_id]
        prediction_query_time = self.prediction_query_time_list[q_t_id]
        reference_time_type = self.C_time_type[in_t_s_id:in_t_e_id, :, :]
        key_time = self.time_list[in_t_s_id:in_t_e_id]
        key_grid = self.time_crime_grid_coor[in_t_s_id:in_t_e_id:, :, :]

        time_type_mask = self.time_type_mask[in_t_s_id:in_t_e_id, :]

        # 为不同的query_time修改mask
        all_time_type_for_query_mask = np.zeros((len(query_time), len(query_time[0]), time_type_mask.shape[0], time_type_mask.shape[1]))
        for q_i in range(len(query_time)):
            query_reference_time = query_time[q_i]
            time_type_for_query_mask = np.zeros((len(query_reference_time), time_type_mask.shape[0], time_type_mask.shape[1]))
            for i in range(len(query_reference_time)):
                q_t = query_reference_time[i]
                id = self._find_id(q_t, key_time)
                if id == len(key_time):
                    time_type_for_query_mask[i] = time_type_mask.copy()
                else:
                    time_type_for_query_mask[i, :id, :] = time_type_mask.copy()[:id, :]
            all_time_type_for_query_mask[q_i] = time_type_for_query_mask


        crime_one_hot = np.zeros(self.n_crime_type)  # 现在要用上了
        if self.predict_crime_id != -1:
            crime_one_hot[self.predict_crime_id] = 1
        grid_feature = self.POI_feature_cate[self._get_grid_id(query_grid)]
        query_grid_feature = np.concatenate((crime_one_hot, grid_feature))

        return reference_time_type, all_time_type_for_query_mask, key_time, key_grid, query_time, np.array(query_grid), \
               query_grid_feature, prediction_query_time, np.array(label)

    def __getitem__(self, item):
        return self._generate_data()


if __name__ == '__main__':
    city_data_info = np.load(('data/NYC/city_base_info_dict_2015_2019.npy'), allow_pickle=True).item()
    # print(city_data_info['n_POI_cate'])
    # print(city_data_info['n_crime_type'])
    with open('data/NYC/train_data/train-crime-0-2015-2019.pkl', 'rb') as f:
        crime_dataset = pickle.load(f)
    f.close()
    bds = BaseDataSet(city_data_info, 10000, crime_dataset, exp_args)
    for it in bds:
        print(it)
    # print(bds)
    pass
