# -*- coding: utf-8 -*-
# @File    : train_one_type_wd.py
# @Author  : juntang
# @Time    : 2023/04/05 14:03

# 所有type一起训练
# 加入点过程
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
from model_v2.model_oh_pp_nowd import crime_prediction_weight_decay
from model_v2.tools.log_tool import log_tool_init
from experiment_v2.NYC.exp_args import exp_args
from data.NYC.data_args import data_args
from model_v2.tools.dataset_utils_oh_pp import BaseDataSet
from model_v2.tools.metrics import precision, recall, f1, get_pre_result, get_score

CUDA_AVAILABLE = False
DEVICE = None


def change():
    exp_args.dataset_size = 200
    exp_args.batch_per_epoch = exp_args.dataset_size // exp_args.batch_size
    pass


def train():  # 原始的多类别训练
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
    DEVICE = torch.device("cuda:{}".format(1) if CUDA_AVAILABLE else "cpu")

    # load data
    crimes = ["THEFT", "ASSAULT 3 & RELATED OFFENSES", "ROBBERY", 'DANGEROUS WEAPONS', 'SEX CRIMES', 'FORGERY']
    # crimes = ['DANGEROUS WEAPONS']
    log_tool_init(model_start_time, "NYC_all_loss_alpha{}".format(exp_args.loss_alpha), exp_args.trained_result,
                  note="con_wd_pp_1day")
    city_data_info = np.load(('data/NYC/city_base_info_dict_2016_2018.npy'), allow_pickle=True).item()
    exp_args.crime_dim = city_data_info['n_crime_type']
    exp_args.grid_feature_dim = city_data_info['n_POI_cate']

    crime_train_dl_list, crime_test_dl_list = [], []  # 现在只train 1day
    for now_crime in crimes:
        now_crime_id = data_args.crime_id[now_crime]
        with open('data/NYC/train_data/train-crime-all-{}-2016-2018.pkl'.format(now_crime_id), 'rb') as f:
            crime_dateset = pickle.load(f)
        f.close()
        with open('data/NYC/pp_data/{}/pp-labels-{}-2016-2018.pkl'.format(exp_args.pp_data_date, now_crime_id),
                  'rb') as f:
            pp_data = pickle.load(f)
        f.close()

        crime_dataset_for_train = crime_dateset[1].copy()  # 取1day
        crime_test_len = math.ceil(len(crime_dataset_for_train[0]) / 8)

        crime_dataset_for_train[0] = crime_dataset_for_train[0][:-crime_test_len]
        crime_dataset_for_test = crime_dateset[1].copy()
        crime_dataset_for_test[0] = crime_dataset_for_test[0][-crime_test_len:]

        train_ds = BaseDataSet(city_base_info=city_data_info, size=exp_args.dataset_size,
                               predict_crime_id=now_crime_id,
                               crime_dataset_all=crime_dataset_for_train, self_adjustment=True, p_ratio=0.2,
                               pp_data=pp_data)
        train_dl = DataLoader(train_ds, batch_size=exp_args.batch_size)

        test_ds = BaseDataSet(city_base_info=city_data_info, size=crime_test_len, predict_crime_id=now_crime_id,
                              crime_dataset_all=crime_dataset_for_test)
        test_dl = DataLoader(test_ds, batch_size=exp_args.batch_size)
        # train_ds.info()
        # test_ds.info()
        crime_train_dl_list.append(train_dl)
        crime_test_dl_list.append(test_dl)

    # crime_prediction_weight_decay and train
    model = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)
    model_meta = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)

    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 主辅任务loss
    cross_loss = nn.CrossEntropyLoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args.lr, weight_decay=exp_args.weight_decay)

    logging.info("Begin training {}".format("NYC"))
    res_TP, res_TN, res_FP, res_FN = [0 for i in range(len(crimes))], \
                                     [0 for i in range(len(crimes))], \
                                     [0 for i in range(len(crimes))], \
                                     [0 for i in range(len(crimes))]
    res_precision, res_recall, res_f1 = [0.0 for i in range(len(crimes))], \
                                        [0.0 for i in range(len(crimes))], \
                                        [0.0 for i in range(len(crimes))]

    for epoch in range(exp_args.train_epoch):
        # train
        for c_id in range(len(crimes)):
            iter_idx = 0
            train_dl = crime_train_dl_list[c_id]
            for it in train_dl:  # batch data
                model.train()
                iter_idx += 1
                reference_time_type, time_type_for_query_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, predicion_time, label, pp_labels = \
                    it[0].to(
                        DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(DEVICE).float(), \
                    it[
                        4].to(
                        DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(DEVICE).float(), \
                    it[
                        8].to(DEVICE), it[9].to(DEVICE).float()

                out, risk_out = model(reference_time_type, time_type_for_query_mask, time_key, time_query, pos_key,
                                      pos_query,
                                      pos_query_feature, prediction_out_concatenate=True)

                # https://www.zhihu.com/question/338559590/answer/2281058020
                # https://arxiv.org/abs/1506.01497
                # 两个损失函数交替训练，这里对于一组测试，可以先用主任务进行loss训练，再用辅任务进行loss训练
                # 但主辅要区分，可以控制辅任务训练的次数，比如几个batch训练一次
                # https://zhuanlan.zhihu.com/p/348873723
                # 多目标优化 多损失函数反向传播

                main_loss = cross_loss(out, label)
                ancillary_loss = mse_loss(risk_out, pp_labels)
                loss = exp_args.loss_alpha * main_loss + (1.0 - exp_args.loss_alpha) * ancillary_loss
                # loss = main_loss
                # TODO: risk_label 拟合辅助任务
                if iter_idx % 10 == 0:
                    p = precision(label, out)
                    r = recall(label, out)
                    f = f1(label, out)
                    logging.info(
                        '|Predict Type {} | Epoch {:02d} / {:04d} | Iter {:04d} / {:04d} | Iter Loss {:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
                            format(c_id, epoch + 1, exp_args.train_epoch, iter_idx, exp_args.batch_per_epoch,
                                   loss.cpu().item(), p, r, f))

                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if iter_idx == exp_args.batch_per_epoch:
                    break

        with torch.no_grad():
            model.eval()
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

                    test_out, _ = model(reference_time_type, time_type_mask, time_key, time_query, pos_key, pos_query,
                                        pos_query_feature, prediction_out_concatenate=True)

                    test_loss_item = cross_loss(test_out, test_label)
                    loss_item.append(test_loss_item.cpu().numpy())
                    TP, TN, FP, FN = get_pre_result(test_label, test_out)
                    aTP, aTN, aFP, aFN = aTP + TP, aTN + TN, aFP + FP, aFN + FN

                mean_test_loss = np.mean(loss_item)
                r_precision, r_recall, r_f1 = get_score(aTP, aTN, aFP, aFN)
                logging.info(
                    '|Predict Type {} | Epoch {:02d} / {:04d} | Test Mean Loss:{:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
                        format(c_id, epoch + 1, exp_args.train_epoch, mean_test_loss,
                               r_precision, r_recall, r_f1))
                logging.info(
                    '|Predict Type {} | Epoch {:02d} / {:04d} | TP:{} | TN:{} | FP:{} | FN:{}'.
                        format(c_id, epoch + 1, exp_args.train_epoch, aTP, aTN, aFP, aFN))

                if r_f1 > res_f1[c_id]:
                    res_precision[c_id], res_recall[c_id], res_f1[c_id] = r_precision, r_recall, r_f1
                    res_TP[c_id], res_TN[c_id], res_FP[c_id], res_FN[c_id] = aTP, aTN, aFP, aFN

    for c_id in range(len(crimes)):
        logging.info(
            'Final Result | Predict Type:{} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f} | TP:{} | TN:{} | FP:{} | FN:{}'.
                format(c_id, res_precision[c_id], res_recall[c_id], res_f1[c_id], res_TP[c_id], res_TN[c_id],
                       res_FP[c_id],
                       res_FN[c_id]))

    model_path = os.path.join(exp_args.trained_model, 'trained_NYC_model_con_{}.pth'.format(model_start_time))
    torch.save(model, model_path)
    pass


def train_test():  # 更改了训练数据的方式，应该更加合理了
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
    DEVICE = torch.device("cuda:{}".format(1) if CUDA_AVAILABLE else "cpu")

    # load data
    crimes = ["THEFT", "ASSAULT 3 & RELATED OFFENSES", "ROBBERY", 'DANGEROUS WEAPONS', 'SEX CRIMES', 'FORGERY']
    # crimes = ["THEFT", "ASSAULT 3 & RELATED OFFENSES"]
    log_tool_init(model_start_time, "NYC_all_TEST_loss_alpha{}".format(exp_args.loss_alpha), exp_args.trained_result,
                  note="no_weight_decay")
    city_data_info = np.load(('data/NYC/city_base_info_dict_2016_2018.npy'), allow_pickle=True).item()
    exp_args.crime_dim = city_data_info['n_crime_type']
    exp_args.grid_feature_dim = city_data_info['n_POI_cate']

    crime_train_dl_list, crime_test_dl_list = [], []  # 现在只train 1day
    for now_crime in crimes:
        now_crime_id = data_args.crime_id[now_crime]
        with open('data/NYC/train_data/train-crime-all-{}-2016-2018.pkl'.format(now_crime_id), 'rb') as f:
            crime_dateset = pickle.load(f)
        f.close()
        with open('data/NYC/pp_data/{}/pp-labels-{}-2016-2018.pkl'.format(exp_args.pp_data_date, now_crime_id),
                  'rb') as f:
            pp_data = pickle.load(f)
        f.close()

        crime_dataset_for_train = crime_dateset[1].copy()  # 取1day
        crime_test_len = math.ceil(len(crime_dataset_for_train[0]) / 8)

        crime_dataset_for_train[0] = crime_dataset_for_train[0][:-crime_test_len]
        crime_dataset_for_test = crime_dateset[1].copy()
        crime_dataset_for_test[0] = crime_dataset_for_test[0][-crime_test_len:]

        train_ds = BaseDataSet(city_base_info=city_data_info, size=exp_args.dataset_size,
                               predict_crime_id=now_crime_id,
                               crime_dataset_all=crime_dataset_for_train, self_adjustment=True, p_ratio=0.2,
                               pp_data=pp_data)
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

    # 主辅任务loss
    cross_loss = nn.CrossEntropyLoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args.lr, weight_decay=exp_args.weight_decay)

    logging.info("Begin training {}".format("NYC"))
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
                reference_time_type, time_type_for_query_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, predicion_time, label, pp_labels = \
                    it[0].to(
                        DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(DEVICE).float(), \
                    it[
                        4].to(
                        DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(DEVICE).float(), \
                    it[
                        8].to(DEVICE), it[9].to(DEVICE).float()

                out, risk_out = model(reference_time_type, time_type_for_query_mask, time_key, time_query, pos_key,
                                      pos_query,
                                      pos_query_feature, prediction_out_concatenate=True)

                main_loss = cross_loss(out, label)
                ancillary_loss = mse_loss(risk_out, pp_labels)
                loss = exp_args.loss_alpha * main_loss + (1.0 - exp_args.loss_alpha) * ancillary_loss
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

                    test_out, _ = model(reference_time_type, time_type_mask, time_key, time_query, pos_key, pos_query,
                                        pos_query_feature, prediction_out_concatenate=True)

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
                                          'trained_NYC_model_con_maf1-{:4f}_{}.pth'.format(all_f / len(crimes),
                                                                                          model_start_time))
                best_all_f = all_f
                # torch.save(model, model_path)

    for c_id in range(len(crimes)):
        logging.info(
            'Final Result | Predict Type:{} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f} | TP:{} | TN:{} | FP:{} | FN:{}'.
                format(data_args.crime_id[crimes[c_id]], res_precision[c_id], res_recall[c_id], res_f1[c_id],
                       res_TP[c_id], res_TN[c_id],
                       res_FP[c_id],
                       res_FN[c_id]))

    # model_path = os.path.join(exp_args.trained_model, 'trained_NYC_model_con_{}.pth'.format(model_start_time))
    # torch.save(model, model_path)

    pass


if __name__ == '__main__':
    # change()
    exp_args.train_epoch = 10
    # exp_args.pp_data_date = '2023-4-26_15-30'  # 一阶的
    exp_args.pp_data_date = '2023-5-10_12-00-o1'  # 二阶的

    train_test()
