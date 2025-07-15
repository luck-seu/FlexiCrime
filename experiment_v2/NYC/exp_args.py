# -*- coding: utf-8 -*-
# @File    : exp_args.py
# @Author  : juntang
# @Time    : 2023/11/09 15:42

from data.NYC.data_args import data_args
import math

class ExperimentArgs:
    def __init__(self):
        self.seed = 123456789
        self.lr = 5e-4
        self.weight_decay = 0.00005

        # train
        self.train_epoch = 20  # 20
        # self.train_epoch = 1  # 20
        # self.batch_per_epoch = 1000  # 500
        self.batch_size = 64  # 20
        self.dataset_size = 20000
        self.batch_per_epoch = self.dataset_size // self.batch_size

        # test
        self.n_test_nums = 1000
        self.test_ratio = 8  # 占1/8


        # crime_prediction_model_v5
        self.freq = 10.0  # 编码地点
        self.time_type_dim = len(data_args.crime_type)  # 类型
        self.crime_dim = 24  # crime 类型数量
        self.grid_feature_dim = 46  # grid特征数量 这是NY的传入，根据city info进行修改
        self.crime_one_hot_hidden = 16
        self.grid_feature_hidden = 32
        self.embed_dim = 64
        self.nhidden = 64  # gru 维度
        self.num_heads = 1
        self.time_pos_ratio = 0.5
        self.query_prediction_time_len = 4
        self.pp_data_date = '2023-4-26_15-30'
        self.loss_alpha = 0.9

        # meta-learning
        self.way = 3
        self.meta_task_size = 20
        self.task_support_num = 5  # support(task train) 5*batch
        self.task_query_num = 1  # query(task test) 1*batch
        self.meta_lr = 5e-4
        self.task_lr = 5e-4
        self.meta_train_epoch = 20
        self.meta_train_batch_num = 500
        self.fine_tunning_epoch = 20
        self.fine_tunning_batch_num = 500
        self.meta_train = ['THEFT', 'ASSAULT 3 & RELATED OFFENSES', 'HARRASSMENT 2', 'CRIMINAL MISCHIEF & RELATED OF', 'ROBBERY', 'BURGLARY']
        self.meta_target = ['DANGEROUS WEAPONS', 'SEX CRIMES', 'FORGERY', 'CRIMINAL TRESPASS']

        # result
        self.trained_model = 'experiment_v2/NYC/train_model'
        self.trained_result = 'experiment_v2/NYC/train_result_log'

        self.rwt_model_start_time = "2023-01-15_16-56"

exp_args = ExperimentArgs()