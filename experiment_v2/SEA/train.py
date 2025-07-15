# -*- coding: utf-8 -*-
# @File    : train_one_type_wd.py
# @Author  : juntang
# @Time    : 2023/12/27 19:03

import math
import os
import time
import torch
import logging
import pickle
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from model_v2.model_v2 import crime_prediction_weight_decay
from model_v2.tools.log_tool import log_tool_init
from experiment_v2.SEA.exp_args import exp_args
from data.Seattle.data_args import data_args
from model_v2.tools.dataset_utils_oh_pp import BaseDataSet
from model_v2.tools.metrics import precision, recall, f1, get_pre_result, get_score

CUDA_AVAILABLE = False
DEVICE = None

if __name__ == '__main__':
    model_start_time = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
    if not os.path.exists(exp_args.trained_model):
        os.makedirs(exp_args.trained_model)
    if not os.path.exists(exp_args.trained_result):
        os.makedirs(exp_args.trained_result)

    # set random seed
    random.seed(exp_args.seed)
    np.random.seed(exp_args.seed)
    torch.manual_seed(exp_args.seed)

    # device choose
    CUDA_AVAILABLE = torch.cuda.is_available()
    print("CUDA_AVAILABLE: ", CUDA_AVAILABLE)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(exp_args.seed)
        N_GPU = torch.cuda.device_count()
    DEVICE = torch.device("cuda:{}".format(0) if CUDA_AVAILABLE else "cpu")

    # load data
    crimes = ['LARCENY-THEFT', 'ASSAULT OFFENSES', 'BURGLARY/BREAKING&ENTERING', 'FRAUD OFFENSES',
              'TRESPASS OF REAL PROPERTY', 'ROBBERY']
    log_tool_init(model_start_time, "SEA_all_first_round", exp_args.trained_result,
                  note="wd_pp_1day")
    city_data_info = np.load(('data/Seattle/city_base_info_dict_2020_2022.npy'), allow_pickle=True).item()
    exp_args.crime_dim = city_data_info['n_crime_type']
    exp_args.grid_feature_dim = city_data_info['n_POI_cate']

    crime_train_dl_list, crime_test_dl_list = [], []  # 现在只train 1day
    for now_crime in crimes:
        now_crime_id = data_args.crime_id[now_crime]
        with open('data/Seattle/train_data/train-crime-all-{}-2020-2022.pkl'.format(now_crime_id), 'rb') as f:
            crime_dateset = pickle.load(f)
        f.close()

        crime_dataset_for_train = crime_dateset[1].copy()  # 取1day
        crime_test_len = math.ceil(len(crime_dataset_for_train[0]) / 8)

        crime_dataset_for_train[0] = crime_dataset_for_train[0][:-crime_test_len]
        crime_dataset_for_test = crime_dateset[1].copy()
        crime_dataset_for_test[0] = crime_dataset_for_test[0][-crime_test_len:]

        train_ds = BaseDataSet(city_base_info=city_data_info, size=exp_args.dataset_size,
                               predict_crime_id=now_crime_id,
                               crime_dataset_all=crime_dataset_for_train, self_adjustment=True, p_ratio=0.2)
        train_dl = DataLoader(train_ds, batch_size=exp_args.batch_size)

        test_ds = BaseDataSet(city_base_info=city_data_info, size=crime_test_len, predict_crime_id=now_crime_id,
                              crime_dataset_all=crime_dataset_for_test)
        test_dl = DataLoader(test_ds, batch_size=exp_args.batch_size)
        crime_train_dl_list.append(train_dl)
        crime_test_dl_list.append(test_dl)

    # crime_prediction_weight_decay and train
    model = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)

    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 任务loss
    cross_loss = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args.lr, weight_decay=exp_args.weight_decay)

    logging.info("Begin training {}".format("SEA"))
    res_TP, res_TN, res_FP, res_FN = [0 for i in range(len(crimes))], \
        [0 for i in range(len(crimes))], \
        [0 for i in range(len(crimes))], \
        [0 for i in range(len(crimes))]
    res_precision, res_recall, res_f1 = [0.0 for i in range(len(crimes))], \
        [0.0 for i in range(len(crimes))], \
        [0.0 for i in range(len(crimes))]

    best_all_f = 0
    for epoch in range(exp_args.train_epoch):
        # train
        for iter_idx in range(1, exp_args.batch_per_epoch + 1):
            for c_id in range(len(crimes)):
                train_dl = crime_train_dl_list[c_id]
                it = next(iter(train_dl))
                model.train()
                reference_time_type, time_type_for_query_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, _, label = \
                    it[0].to(
                        DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(
                        DEVICE).float(), \
                        it[4].to(
                            DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(
                        DEVICE).float(), \
                        it[8].to(DEVICE)

                out = model(reference_time_type, time_type_for_query_mask, time_key, time_query, pos_key,
                            pos_query,
                            pos_query_feature, prediction_out_concatenate=False)

                loss = cross_loss(out, label)
                # loss = main_loss
                if iter_idx % 10 == 0:
                    p = precision(label, out)
                    r = recall(label, out)
                    f = f1(label, out)
                    crime_type = data_args.crime_id[crimes[c_id]]
                    logging.info(
                        '|Predict Type {} | Epoch {:02d} / {:04d} | Iter {:04d} / {:04d} | Iter Loss {:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
                        format(crime_type, epoch + 1, exp_args.train_epoch, iter_idx, exp_args.batch_per_epoch,
                               loss.cpu().item(), p, r, f))

                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            model.eval()
            all_f = 0
            for c_id in range(len(crimes)):
                test_dl = crime_test_dl_list[c_id]
                loss_item = []
                aTP, aTN, aFP, aFN = 0, 0, 0, 0
                for it in test_dl:
                    reference_time_type, time_type_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, _, test_label = \
                        it[0].to(
                            DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(
                            DEVICE).float(), \
                            it[4].to(
                                DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(
                            DEVICE).float(), \
                            it[8].to(DEVICE)

                    test_out = model(reference_time_type, time_type_mask, time_key, time_query, pos_key, pos_query,
                                     pos_query_feature, prediction_out_concatenate=False)

                    test_loss_item = cross_loss(test_out, test_label)
                    loss_item.append(test_loss_item.cpu().numpy())
                    TP, TN, FP, FN = get_pre_result(test_label, test_out)
                    aTP, aTN, aFP, aFN = aTP + TP, aTN + TN, aFP + FP, aFN + FN

                mean_test_loss = np.mean(loss_item)
                r_precision, r_recall, r_f1 = get_score(aTP, aTN, aFP, aFN)
                crime_type = data_args.crime_id[crimes[c_id]]
                logging.info(
                    '|Predict Type {} | Epoch {:02d} / {:04d} | Test Mean Loss:{:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
                    format(crime_type, epoch + 1, exp_args.train_epoch, mean_test_loss,
                           r_precision, r_recall, r_f1))
                logging.info(
                    '|Predict Type {} | Epoch {:02d} / {:04d} | TP:{} | TN:{} | FP:{} | FN:{}'.
                    format(crime_type, epoch + 1, exp_args.train_epoch, aTP, aTN, aFP, aFN))
                all_f += r_f1
                if r_f1 > res_f1[c_id]:
                    res_precision[c_id], res_recall[c_id], res_f1[c_id] = r_precision, r_recall, r_f1
                    res_TP[c_id], res_TN[c_id], res_FP[c_id], res_FN[c_id] = aTP, aTN, aFP, aFN

            if all_f > best_all_f:
                model_path = os.path.join(exp_args.trained_model,
                                          'trained_NYC_model_con_{}_maf1-{:4f}.pth'.format(model_start_time,
                                                                                           all_f / len(crimes)
                                                                                           ))
                best_all_f = all_f
                torch.save(model, model_path)

    for c_id in range(len(crimes)):
        logging.info(
            'Final Result | Predict Type:{} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f} | TP:{} | TN:{} | FP:{} | FN:{}'.
            format(data_args.crime_id[crimes[c_id]], res_precision[c_id], res_recall[c_id], res_f1[c_id],
                   res_TP[c_id], res_TN[c_id],
                   res_FP[c_id],
                   res_FN[c_id]))
