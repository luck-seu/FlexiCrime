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
    DEVICE = torch.device("cuda:{}".format(1) if CUDA_AVAILABLE else "cpu")

    # load data
    crimes = ["THEFT", "ASSAULT 3 & RELATED OFFENSES", "ROBBERY", 'DANGEROUS WEAPONS', 'SEX CRIMES', 'FORGERY']
    # log_tool_init(model_start_time, "NYC_all_first_round", exp_args.trained_result,
    #               note="wd_pp_1day")
    city_data_info = np.load(('data/NYC/city_base_info_dict_2016_2018.npy'), allow_pickle=True).item()
    exp_args.crime_dim = city_data_info['n_crime_type']
    exp_args.grid_feature_dim = city_data_info['n_POI_cate']

    crime_train_dl_list, crime_test_dl_list = [], []  #
    for now_crime in crimes:
        now_crime_id = data_args.crime_id[now_crime]
        with open('data/NYC/train_data/case-crime-{}-2016-2018.pkl'.format(now_crime_id), 'rb') as f:
            crime_dataset = pickle.load(f)
        f.close()
        # with open('data/NYC/pp_data/{}/pp-labels-{}-2016-2018.pkl'.format(exp_args.pp_data_date, now_crime_id),
        #           'rb') as f:
        #     pp_data = pickle.load(f)
        # f.close()
        case_dataset = crime_dataset[0].copy()
        test_ds = BaseDataSet(city_base_info=city_data_info, size=len(case_dataset[0]), predict_crime_id=now_crime_id,
                              crime_dataset_all=case_dataset, case=True)
        test_dl = DataLoader(test_ds, batch_size=exp_args.batch_size)
        crime_test_dl_list.append(test_dl)

    model_path = '/home/tangjun/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_NYC_model_con_maf1-0.356194_2023-11-21_16-16.pth'
    model = torch.load(model_path)
    model.device = DEVICE  # 重新设置模型里的device
    if CUDA_AVAILABLE:
        model = model.to(DEVICE)
    print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    with torch.no_grad():
        model.eval()
        for c_id in range(len(crimes)):
            time_value_list, pos_value_list = [], []
            now_dl = crime_test_dl_list[c_id]
            for it in now_dl:
                reference_time_type, time_type_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, _, test_label = \
                    it[0].to(
                        DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(
                        DEVICE).float(), \
                        it[4].to(
                            DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(
                        DEVICE).float(), \
                        it[8].to(DEVICE)

                time_value, pos_value = model(reference_time_type, time_type_mask, time_key, time_query, pos_key, pos_query,
                                    pos_query_feature, prediction_out_concatenate=False, get_crime_context=True)

                tv = time_value.squeeze().cpu().tolist()
                pv = pos_value.squeeze().cpu().tolist()
                time_value_list.extend(tv)
                pos_value_list.extend(pv)

            res = [time_value_list, pos_value_list]
            write_name = 'case-feature-crime-{}-{}-{}.pkl'.format(data_args.crime_id[crimes[c_id]],
                                                          data_args.crime_use_data[0],
                                                          data_args.crime_use_data[1])
            with open(os.path.join('experiment_v2/NYC/train_data', write_name), 'wb') as f:
                pickle.dump(res, f)
            f.close()


                # test_loss_item = cross_loss(test_out, test_label)
                # loss_item.append(test_loss_item.cpu().numpy())
                # TP, TN, FP, FN = get_pre_result(test_label, test_out)
                # aTP, aTN, aFP, aFN = aTP + TP, aTN + TN, aFP + FP, aFN + FN

            # mean_test_loss = np.mean(loss_item)
            # r_precision, r_recall, r_f1 = get_score(aTP, aTN, aFP, aFN)
            # r_acc = (aTP + aTN) / (aTP + aTN + aFP + aFN)
            # logging.info(
            #     '|Predict Type {} | Diff Type {} | Test Mean Loss:{:.8f} | Precision:{:.4f} | Recall:{:.4f} | Acc:{:.4f} | F1:{:.4f}'.
            #     format(c_id, id, mean_test_loss, r_precision, r_recall, r_acc, r_f1))
            # logging.info(
            #     '|Predict Type {} | Diff Type {} | TP:{} | TN:{} | FP:{} | FN:{}'.
            #     format(c_id, id, aTP, aTN, aFP, aFN))

    pass
