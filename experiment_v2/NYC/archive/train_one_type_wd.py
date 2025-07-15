# -*- coding: utf-8 -*-
# @File    : train_one_type_wd.py
# @Author  : juntang
# @Time    : 2023/11/9 21:26


# -*- coding: utf-8 -*-
# @File    : train_one_type_wd.py
# @Author  : juntang
# @Time    : 2023/01/11 14:03
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
from experiment_v2.NYC.exp_args import exp_args
from data.NYC.data_args import data_args
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
    now_crime = 'THEFT'
    now_crime_id = data_args.crime_id[now_crime]
    log_tool_init(model_start_time, now_crime_id, exp_args.trained_result, note="wd_1day")

    city_data_info = np.load(('data/NYC/city_base_info_dict_2016_2018.npy'), allow_pickle=True).item()
    exp_args.crime_dim = city_data_info['n_crime_type']
    exp_args.grid_feature_dim = city_data_info['n_POI_cate']
    with open('data/NYC/train_data/train-crime-all-{}-2016-2018.pkl'.format(now_crime_id), 'rb') as f:
        crime_dateset = pickle.load(f)
    f.close()
    # 12h 1day 3day 5day，现只train 1day
    train_dl_list, test_dl_list = [], []
    crime_dataset_type_len = len(crime_dateset)
    for i in range(0, 2):
        crime_dataset_for_train = crime_dateset[i].copy()
        crime_test_len = math.ceil(len(crime_dataset_for_train[0]) / 8)

        crime_dataset_for_train[0] = crime_dataset_for_train[0][:-crime_test_len]
        crime_dataset_for_test = crime_dateset[i].copy()
        crime_dataset_for_test[0] = crime_dataset_for_test[0][-crime_test_len:]

        train_ds = BaseDataSet(city_base_info=city_data_info, size=exp_args.dataset_size,  # ,size=len(crime_dataset_for_train[0]),
                               crime_dataset_all=crime_dataset_for_train, self_adjustment=True, p_ratio=0.1)
        train_dl = DataLoader(train_ds, batch_size=exp_args.batch_size)

        test_ds = BaseDataSet(city_base_info=city_data_info, size=crime_test_len,
                              crime_dataset_all=crime_dataset_for_test)
        test_dl = DataLoader(test_ds, batch_size=exp_args.batch_size)
        # train_ds.info()
        # test_ds.info()
        train_dl_list.append(train_dl)
        test_dl_list.append(test_dl)

    # crime_prediction_weight_decay and train
    model = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)
    model_meta = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)

    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args.lr, weight_decay=exp_args.weight_decay)

    logging.info("Begin training {}".format(now_crime))
    res_TP, res_TN, res_FP, res_FN = [0 for i in range(crime_dataset_type_len)], \
                                     [0 for i in range(crime_dataset_type_len)], \
                                     [0 for i in range(crime_dataset_type_len)], \
                                     [0 for i in range(crime_dataset_type_len)]
    res_precision, res_recall, res_f1 = [0.0 for i in range(crime_dataset_type_len)], \
                                        [0.0 for i in range(crime_dataset_type_len)], \
                                        [0.0 for i in range(crime_dataset_type_len)]

    # exp_args.batch_per_epoch = 1000
    exp_args.train_epoch = 30
    for epoch in range(exp_args.train_epoch):
        # train
        epoch_loss = [0 for i in range(crime_dataset_type_len)]
        for tp in range(1, 2):
            iter_idx = 0
            # epoch_loss = 0
            train_dl = train_dl_list[tp]
            for it in train_dl:  # batch data
                model.train()
                iter_idx += 1
                reference_time_type, time_type_for_query_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, predicion_time, label = \
                it[0].to(
                    DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(DEVICE).float(), it[
                    4].to(
                    DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(DEVICE).float(), it[
                    8].to(DEVICE)

                out = model(reference_time_type, time_type_for_query_mask, time_key, time_query, pos_key, pos_query,
                            pos_query_feature)

                loss = criterion(out, label)
                epoch_loss[tp] += loss.cpu().item()
                if iter_idx % 10 == 0:
                    p = precision(label, out)
                    r = recall(label, out)
                    f = f1(label, out)
                    logging.info(
                        '|Predict Type {} | Epoch {:02d} / {:04d} | Iter {:04d} / {:04d} | Iter Loss {:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
                            format(tp, epoch + 1, exp_args.train_epoch, iter_idx, exp_args.batch_per_epoch,
                                   loss.cpu().item(), p, r, f))

                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter_idx == exp_args.batch_per_epoch:
                    break


        with torch.no_grad():
            model.eval()
            for tp in range(1, 2):
                test_dl = test_dl_list[tp]
                precision_epoch_item, recall_epoch_item, f1_epoch_item = [], [], []
                loss_item = []
                aTP, aTN, aFP, aFN = 0, 0, 0, 0
                for it in test_dl:
                    reference_time_type, time_type_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, _, test_label = \
                    it[0].to(
                        DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(DEVICE).float(), \
                    it[4].to(
                        DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(DEVICE).float(), \
                    it[8].to(DEVICE)

                    test_out = model(reference_time_type, time_type_mask, time_key, time_query, pos_key, pos_query,
                                     pos_query_feature)

                    test_loss_item = criterion(test_out, test_label)
                    loss_item.append(test_loss_item.cpu().numpy())
                    # if 1 in test_label:
                    #     precision_epoch_item.append(precision(test_label, test_out))
                    #     recall_epoch_item.append(recall(test_label, test_out))
                    #     f1_epoch_item.append(f1(test_label, test_out))
                    TP, TN, FP, FN = get_pre_result(test_label, test_out)
                    aTP, aTN, aFP, aFN = aTP + TP, aTN + TN, aFP + FP, aFN + FN

                mean_test_loss = np.mean(loss_item)
                # mean_precision = np.mean(precision_epoch_item)
                # mean_recall = np.mean(recall_epoch_item)
                # mean_f1 = np.mean(f1_epoch_item)
                r_precision, r_recall, r_f1 = get_score(aTP, aTN, aFP, aFN)
                logging.info(
                    '|Predict Type {} | Epoch {:02d} / {:04d} | Epoch Loss :{:8f} | Test Mean Loss:{:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
                        format(tp, epoch + 1, exp_args.train_epoch, epoch_loss[tp] / exp_args.batch_per_epoch,
                               mean_test_loss,
                               r_precision, r_recall, r_f1))
                logging.info(
                    '|Predict Type {} | Epoch {:02d} / {:04d} | TP:{} | TN:{} | FP:{} | FN:{}'.
                        format(tp, epoch + 1, exp_args.train_epoch, aTP, aTN, aFP, aFN))

                if r_f1 > res_f1[tp]:
                    res_precision[tp], res_recall[tp], res_f1[tp] = r_precision, r_recall, r_f1
                    res_TP[tp], res_TN[tp], res_FP[tp], res_FN[tp] = aTP, aTN, aFP, aFN

    for tp in range(1, 2):
        logging.info(
            'Final Result | Predict Type:{} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f} | TP:{} | TN:{} | FP:{} | FN:{}'.
            format(tp, res_precision[tp], res_recall[tp], res_f1[tp], res_TP[tp], res_TN[tp], res_FP[tp], res_FN[tp]))

    # save model_v3
    # model_path = os.path.join(exp_args.trained_model,
    #                           'trained_crime-{}_model_wd_{}.pth'.format(now_crime_id, model_start_time))
    # torch.save(model, model_path)

    pass
