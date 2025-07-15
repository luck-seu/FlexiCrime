# -*- coding: utf-8 -*-
# @File    : train_one_type_wd.py
# @Author  : juntang
# @Time    : 2023/12/10 16:26

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
from model_v2.tools.dataset_utils_oh_pp import FineTuningBaseDataSet
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
    DEVICE = torch.device("cuda:{}".format(1) if CUDA_AVAILABLE else "cpu")

    # load data
    # 'THEFT': 0, 'ASSAULT 3 & RELATED OFFENSES': 1, 'ROBBERY': 6, 'DANGEROUS WEAPONS': 10, 'SEX CRIMES': 12, 'FORGERY': 13
    now_crime = 'DANGEROUS WEAPONS'
    now_crime_id = data_args.crime_id[now_crime]
    log_tool_init(model_start_time, now_crime_id, exp_args.trained_result, note="different_slot")

    # 装载模型
    # /home/tangjun/CrimePrediction_V2/experiment_v2/NYC/train_model/trained_0_fine_tuning_2023-12-19_20-37_f1-0.866227.pth
    # /home/tangjun/CrimePrediction_V2/experiment_v2/NYC/train_model/trained_1_fine_tuning_2023-12-20_15-42_f1-0.584906.pth
    # /home/tangjun/CrimePrediction_V2/experiment_v2/NYC/train_model/trained_6_fine_tuning_2023-12-23_14-34_f1-0.215171.pth
    # /home/tangjun/CrimePrediction_V2/experiment_v2/NYC/train_model/trained_10_fine_tuning_2023-12-21_13-32_f1-0.234875.pth
    # /home/tangjun/CrimePrediction_V2/experiment_v2/NYC/train_model/trained_12_fine_tuning_2023-12-21_17-59_f1-0.365157.pth
    # /home/tangjun/CrimePrediction_V2/experiment_v2/NYC/train_model/trained_13_fine_tuning_2023-12-21_20-57_f1-0.243243.pth

    model_path = '/home/tangjun/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_10_fine_tuning_2023-12-19_21-12_f1-0.118519.pth'
    model = torch.load(model_path)
    model.device = DEVICE

    city_data_info = np.load(('data/NYC/city_base_info_dict_2016_2018.npy'), allow_pickle=True).item()
    exp_args.crime_dim = city_data_info['n_crime_type']
    exp_args.grid_feature_dim = city_data_info['n_POI_cate']
    with open('data/NYC/train_data/crime-different-slot-ownid-{}-2016-2018.pkl'.format(now_crime_id), 'rb') as f:
        crime_dateset = pickle.load(f)
    f.close()

    # 进行测试

    train_dl_list, test_dl_list = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    crime_dataset_type_len = len(crime_dateset)
    hour_list = [4, 6, 8, 12, 24]
    for i in range(len(hour_list)):
        pp_feature_dataset_name = '/home/tangjun/CrimePrediction_LAB_WIN/neural_stpp/train_data/NYC/0/crime-pp-feature-{}hours.pkl'.format(
            hour_list[i])
        with open(pp_feature_dataset_name, 'rb') as f:
            crime_pp_feature_dataset = pickle.load(f)
        f.close()

        crime_dataset_for_train = crime_dateset[i].copy()
        crime_test_len = math.ceil(len(crime_dataset_for_train[0]) / 8)

        crime_dataset_for_train[0] = crime_dataset_for_train[0][:-crime_test_len]
        crime_dataset_for_test = crime_dateset[i].copy()
        crime_dataset_for_test[0] = crime_dataset_for_test[0][-crime_test_len:]

        train_ds = FineTuningBaseDataSet(city_base_info=city_data_info, size=exp_args.dataset_size,
                                         crime_dataset_all=crime_dataset_for_train, pp_feature=crime_pp_feature_dataset,
                                         self_adjustment=True, p_ratio=0.1)
        train_dl = DataLoader(train_ds, batch_size=exp_args.batch_size)

        test_ds = FineTuningBaseDataSet(city_base_info=city_data_info, size=crime_test_len,
                                        crime_dataset_all=crime_dataset_for_test, pp_feature=crime_pp_feature_dataset)
        test_dl = DataLoader(test_ds, batch_size=exp_args.batch_size)
        # train_ds.info()
        # test_ds.info()
        train_dl_list[i] = train_dl
        test_dl_list[i] = test_dl
        # train_dl_list.append(train_dl)
        # test_dl_list.append(test_dl)

    # model = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)

    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args.lr, weight_decay=exp_args.weight_decay)

    logging.info("Begin Testing {}".format(now_crime))
    res_TP, res_TN, res_FP, res_FN = [0 for i in range(crime_dataset_type_len)], \
        [0 for i in range(crime_dataset_type_len)], \
        [0 for i in range(crime_dataset_type_len)], \
        [0 for i in range(crime_dataset_type_len)]
    res_precision, res_recall, res_f1 = [0.0 for i in range(crime_dataset_type_len)], \
        [0.0 for i in range(crime_dataset_type_len)], \
        [0.0 for i in range(crime_dataset_type_len)]

    # for test
    with torch.no_grad():
        model.eval()
        for tp in range(len(hour_list)):
            test_dl = test_dl_list[tp]
            precision_epoch_item, recall_epoch_item, f1_epoch_item = [], [], []
            loss_item = []
            aTP, aTN, aFP, aFN = 0, 0, 0, 0
            for it in test_dl:
                reference_time_type, time_type_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, _, test_label, pp_feature = \
                    it[0].to(
                        DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(
                        DEVICE).float(), \
                        it[4].to(
                            DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(
                        DEVICE).float(), it[8].to(DEVICE), it[9].to(DEVICE).float()

                test_out = model(reference_time_type, time_type_mask, time_key, time_query, pos_key, pos_query,
                                 pos_query_feature, prediction_out_concatenate=True, short_term_feature=pp_feature)

                test_loss_item = criterion(test_out, test_label)
                loss_item.append(test_loss_item.cpu().numpy())
                TP, TN, FP, FN = get_pre_result(test_label, test_out)
                aTP, aTN, aFP, aFN = aTP + TP, aTN + TN, aFP + FP, aFN + FN

            mean_test_loss = np.mean(loss_item)
            r_precision, r_recall, r_f1 = get_score(aTP, aTN, aFP, aFN)

            # logging.info(
            #     '|Predict {} Hours| Epoch {:02d} / {:04d} | Epoch Loss :{:8f} | Test Mean Loss:{:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
            #     format(hour_list[tp], epoch + 1, exp_args.train_epoch, epoch_loss[tp] / exp_args.batch_per_epoch,
            #            mean_test_loss,
            #            r_precision, r_recall, r_f1))
            # logging.info(
            #     '|Predict {} Hours| Epoch {:02d} / {:04d} | TP:{} | TN:{} | FP:{} | FN:{}'.
            #     format(hour_list[tp], epoch + 1, exp_args.train_epoch, aTP, aTN, aFP, aFN))
            print(hour_list[tp], 'ok')
            if r_f1 > res_f1[tp]:
                res_precision[tp], res_recall[tp], res_f1[tp] = r_precision, r_recall, r_f1
                res_TP[tp], res_TN[tp], res_FP[tp], res_FN[tp] = aTP, aTN, aFP, aFN

                # model_path = os.path.join(exp_args.trained_model,
                #                           'trained_{}_fine_tuning_{}_f1-{:4f}.pth'.format(now_crime_id, model_start_time, res_f1[tp]
                #                                                                          ))
                # torch.save(model, model_path)
                # best_all_f = all_f
                # torch.save(model, model_path)

    for tp in range(len(hour_list)):
        logging.info(
            'Final Result | Predict {} Hours | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f} | TP:{} | TN:{} | FP:{} | FN:{}'.
            format(hour_list[tp], res_precision[tp], res_recall[tp], res_f1[tp], res_TP[tp], res_TN[tp], res_FP[tp], res_FN[tp]))



    pass
