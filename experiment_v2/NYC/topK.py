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
    DEVICE = torch.device("cuda:{}".format(0) if CUDA_AVAILABLE else "cpu")
    # DEVICE = torch.device("cpu")

    interval_choose = 4
    now_crimes = ['THEFT', 'ASSAULT 3 & RELATED OFFENSES', 'ROBBERY', 'DANGEROUS WEAPONS', 'SEX CRIMES', 'FORGERY']
    model_paths = [
        '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_0_fine_tuning_2023-12-19_20-37_f1-0.866227.pth',
        '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_1_fine_tuning_2023-12-20_15-42_f1-0.584906.pth',
        '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_6_fine_tuning_2024-01-12_11-56_f1-0.212329.pth',
        '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_10_fine_tuning_2024-01-12_11-57_f1-0.214545.pth',
        '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_NYC_model_con_maf1-0.361709_2023-12-14_19-53.pth',
        '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_NYC_model_con_maf1-0.361709_2023-12-14_19-53.pth'
    ]
    for crime_type in range(6):
        # load data
        # now_crime = 'THEFT'
        # now_crime = 'ASSAULT 3 & RELATED OFFENSES'
        # now_crime = 'ROBBERY'
        # now_crime = 'DANGEROUS WEAPONS'
        # now_crime = 'SEX CRIMES'
        # now_crime = 'FORGERY'
        now_crime = now_crimes[crime_type]
        now_crime_id = data_args.crime_id[now_crime]
        log_tool_init(model_start_time, now_crime_id, exp_args.trained_result, note="fine_tuning")

        city_data_info = np.load('/home/hongyi/CrimePrediction_LAB_WIN/data/NYC/city_base_info_dict_2016_2018.npy',
                                 allow_pickle=True).item()
        exp_args.crime_dim = city_data_info['n_crime_type']
        exp_args.grid_feature_dim = city_data_info['n_POI_cate']
        with open(
                '/home/hongyi/CrimePrediction_LAB_WIN/data/NYC/train_data/crime-different-slot-ownid-{}-2016-2018.pkl'.format(
                    now_crime_id), 'rb') as f:
            crime_dateset = pickle.load(f)
        f.close()

        with open(
                '/home/hongyi/CrimePrediction_LAB_WIN/data/NYC/train_data/crime-different-slot-ownid-{}-2016-2018-CrimeNumCount.pkl'.format(
                    now_crime_id), 'rb') as f:
            crime_dateset_count = pickle.load(f)
        f.close()

        # 微调只train 24hours ?
        train_dl_list, test_dl_list = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        crime_dataset_type_len = len(crime_dateset)
        hour_list = [4, 6, 8, 12, 24]
        for i in range(interval_choose, interval_choose + 1):
            pp_feature_dataset_name = '/home/hongyi/CrimePrediction_LAB_WIN/neural_stpp/train_data/NYC/{}/crime-pp-feature-{}hours.pkl'.format(
                now_crime_id,
                hour_list[i])
            with open(pp_feature_dataset_name, 'rb') as f:
                crime_pp_feature_dataset = pickle.load(f)
            f.close()

            crime_dataset_for_train = crime_dateset[i].copy()
            crime_test_len = math.ceil(len(crime_dataset_for_train[0]) / 8)
            degree = crime_test_len // 42
            crime_test_len = degree * 42
            crime_dataset_for_train[0] = crime_dataset_for_train[0][:-crime_test_len]

            crime_dataset_for_test = crime_dateset[i].copy()
            crime_dataset_for_test[0] = crime_dataset_for_test[0][-crime_test_len:]
            crime_dataset_for_test_count = crime_dateset_count[i].copy()
            crime_dataset_for_test_count[0] = crime_dataset_for_test_count[0][-crime_test_len:]

            train_ds = FineTuningBaseDataSet(city_base_info=city_data_info, size=exp_args.dataset_size,
                                             crime_dataset_all=crime_dataset_for_train,
                                             pp_feature=crime_pp_feature_dataset,
                                             self_adjustment=True, p_ratio=0.15)
            train_dl = DataLoader(train_ds, batch_size=exp_args.batch_size)

            test_ds = FineTuningBaseDataSet(city_base_info=city_data_info, size=crime_test_len,
                                            crime_dataset_all=crime_dataset_for_test,
                                            crime_dataset_all_count=crime_dataset_for_test_count,
                                            pp_feature=crime_pp_feature_dataset)
            test_dl = DataLoader(test_ds, batch_size=exp_args.batch_size)
            # train_ds.info()
            # test_ds.info()
            train_dl_list[i] = train_dl
            test_dl_list[i] = test_dl
            # train_dl_list.append(train_dl)
            # test_dl_list.append(test_dl)

        # crime_prediction_weight_decay and train
        # model_path = '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_NYC_model_con_maf1-0.361709_2023-12-14_19-53.pth'
        # model_path = '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_0_fine_tuning_2023-12-19_20-37_f1-0.866227.pth'
        # model_path = '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_1_fine_tuning_2023-12-20_15-42_f1-0.584906.pth'
        # model_path = '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_6_fine_tuning_2024-01-12_11-56_f1-0.212329.pth'
        # model_path = '/home/hongyi/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_model/trained_10_fine_tuning_2024-01-12_11-57_f1-0.214545.pth'
        model_path = model_paths[crime_type]
        model = torch.load(model_path)
        model.device = DEVICE  # 重新设置模型里的device
        # model.to(DEVICE)
        # model = crime_prediction_weight_decay(exp_args=exp_args, device=DEVICE)

        if CUDA_AVAILABLE:
            model = model.to(DEVICE)
        print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp_args.lr, weight_decay=exp_args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        logging.info("Begin training {}".format(now_crime))
        res_TP, res_TN, res_FP, res_FN = [0 for i in range(crime_dataset_type_len)], \
            [0 for i in range(crime_dataset_type_len)], \
            [0 for i in range(crime_dataset_type_len)], \
            [0 for i in range(crime_dataset_type_len)]
        res_precision, res_recall, res_f1 = [0.0 for i in range(crime_dataset_type_len)], \
            [0.0 for i in range(crime_dataset_type_len)], \
            [0.0 for i in range(crime_dataset_type_len)]

        # exp_args.batch_per_epoch = 1000
        # exp_args.train_epoch = 30

        # for epoch in range(exp_args.train_epoch):
        for epoch in range(1):
            # train
            epoch_loss = [0 for i in range(crime_dataset_type_len)]

            # for tp in range(4, 5):
            #     iter_idx = 0
            #     # epoch_loss = 0
            #     train_dl = train_dl_list[tp]
            #     for it in train_dl:  # batch data
            #         model.train()
            #         iter_idx += 1
            #         reference_time_type, time_type_for_query_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, predicion_time, label, pp_feature = \
            #             it[0].to(
            #                 DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(DEVICE).float(), \
            #                 it[
            #                     4].to(
            #                     DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(
            #                 DEVICE).float(), \
            #                 it[
            #                     8].to(DEVICE), it[9].to(DEVICE).float()
            #
            #         out = model(reference_time_type, time_type_for_query_mask, time_key, time_query, pos_key, pos_query,
            #                     pos_query_feature, prediction_out_concatenate=True, short_term_feature=pp_feature)
            #
            #         loss = criterion(out, label)
            #         epoch_loss[tp] += loss.cpu().item()
            #         if iter_idx % 10 == 0:
            #             p = precision(label, out)
            #             r = recall(label, out)
            #             f = f1(label, out)
            #             logging.info(
            #                 '|Predict Type {} | Epoch {:02d} / {:04d} | Iter {:04d} / {:04d} | Iter Loss {:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
            #                 format(now_crime_id, epoch + 1, exp_args.train_epoch, iter_idx, exp_args.batch_per_epoch,
            #                        loss.cpu().item(), p, r, f))
            #
            #         # 更新
            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
            #
            #         if iter_idx == exp_args.batch_per_epoch:
            #             break

            # # scheduler.step()

            out_res0 = np.array([])
            out_res1 = np.array([])
            test_lables = np.array([])
            test_lables_count = np.array([])
            time_costs = []
            with (torch.no_grad()):
                model.eval()
                for tp in range(interval_choose, interval_choose + 1):
                    test_dl = test_dl_list[tp]
                    precision_epoch_item, recall_epoch_item, f1_epoch_item = [], [], []
                    loss_item = []
                    aTP, aTN, aFP, aFN = 0, 0, 0, 0
                    for it in test_dl:
                        reference_time_type, time_type_mask, time_key, pos_key, time_query, pos_query, pos_query_feature, _, test_label, pp_feature = \
                            it[0].to(DEVICE).float(), it[1].to(DEVICE).float(), it[2].to(DEVICE).float(), it[3].to(
                                DEVICE).float(), \
                                it[4].to(DEVICE).float(), it[5].to(DEVICE).float(), it[6].to(DEVICE).float(), it[7].to(
                                DEVICE).float(), \
                                it[8].to(DEVICE), it[9].to(DEVICE).float()
                        test_label_count = it[10].to(DEVICE)

                        start = time.time()
                        test_out = model(reference_time_type, time_type_mask, time_key, time_query, pos_key, pos_query,
                                         pos_query_feature, prediction_out_concatenate=True,
                                         short_term_feature=pp_feature)
                        end = time.time()
                        cost = end - start
                        # print('time cost : %.5f sec' %cost)
                        time_costs.append(cost)

                        out_res0 = np.append(out_res0, test_out[:, 0].cpu().numpy())
                        out_res1 = np.append(out_res1, test_out[:, 1].cpu().numpy())
                        test_lables = np.append(test_lables, test_label.cpu().numpy())
                        test_lables_count = np.append(test_lables_count, test_label_count.cpu().numpy())

                        test_loss_item = criterion(test_out, test_label)
                        loss_item.append(test_loss_item.cpu().numpy())
                        TP, TN, FP, FN = get_pre_result(test_label, test_out)
                        aTP, aTN, aFP, aFN = aTP + TP, aTN + TN, aFP + FP, aFN + FN

                    len_out_res = len(out_res0)
                    len_test_lables = len(test_lables)
                    out_res0 = [out_res0[i:i + 42] for i in
                                range(0, len_out_res, 42)]
                    out_res1 = [out_res1[i:i + 42] for i in
                                range(0, len_out_res, 42)]
                    test_lables = [test_lables[i:i + 42] for i in
                                   range(0, len_test_lables, 42)]
                    test_lables_count = [test_lables_count[i:i + 42] for i in
                                         range(0, len_test_lables, 42)]
                    hrs = np.array([])
                    k = 10
                    # for out0, out1, test_lable in zip(out_res0, out_res1, test_lables):
                    #     min_k_indices = np.argpartition(out0, k)[:k]
                    #     max_k_indices = np.argpartition(out1, -k)[-k:]
                    #     truth_k0 = test_lable[min_k_indices]
                    #     truth_k1 = test_lable[max_k_indices]
                    #     Hit = 0
                    #     for truth_0, truth_1 in zip(truth_k0, truth_k1):
                    #         if truth_0 == 1 or truth_1 == 1:
                    #             Hit += 1
                    #     GT = np.sum(test_lable)
                    #     if GT > 0:
                    #         # hr = Hit / GT
                    #         # hrs = np.append(hrs, hr)
                    #         hrs = np.append(hrs, Hit)
                    #
                    # truthk_max = len(hrs) * 10
                    # if truthk_max >= np.sum(test_lables):
                    #     HR = np.sum(hrs) / np.sum(test_lables)
                    # else:
                    #     HR = np.sum(hrs) / truthk_max

                    IDCG = sum([1 / np.log2(i + 2) for i in range(k)])
                    hrs = []
                    gts = []
                    aps = []
                    ndcgs = []
                    for out0, out1, test_lable, test_lable_count in zip(out_res0, out_res1, test_lables,
                                                                        test_lables_count):
                        min_pred0_k_index = np.argsort(out0)[:k]
                        max_pred1_k_index = np.argsort(out1)[:-(k + 1):-1]
                        true_k_indices = np.argsort(test_lable_count)[:-(k + 1):-1]

                        # 计算Hit Ratio
                        Hit = 0
                        for HR_index in true_k_indices:
                            if HR_index in min_pred0_k_index or HR_index in max_pred1_k_index:
                                Hit += 1
                        GT = np.sum(test_lable[true_k_indices])
                        if GT > 0:
                            hrs.append(Hit)
                            gts.append(GT)

                        # 计算Average Precision
                        precisions = []
                        rank = 1
                        for PRE_index in range(0, k, 1):
                            if test_lable[min_pred0_k_index[PRE_index]] == 1 or test_lable[
                                max_pred1_k_index[PRE_index]] == 1:
                                Precision = rank / (PRE_index + 1)
                                rank += 1
                                precisions.append(Precision)
                        precisions_num = len(precisions)
                        if precisions_num > 0:
                            AP = sum(precisions) / precisions_num
                            aps.append(AP)

                        # 计算Normalized Discounted Cummulative Gain
                        cgs = []
                        for DCG_index in range(0, k, 1):
                            if test_lable[min_pred0_k_index[DCG_index]] == 1 or test_lable[
                                max_pred1_k_index[DCG_index]] == 1:
                                CG = 1 / np.log2(DCG_index + 2)
                                cgs.append(CG)
                        cgs_num = len(cgs)
                        if cgs_num > 0:
                            DCG = sum(cgs)
                            NDCG = DCG / IDCG
                            ndcgs.append(NDCG)

                    time_cost_average = sum(time_costs) / len(out_res0)
                    print(time_cost_average)
                    test_lables_count_std_num = sum(gts) if sum(gts) > sum(hrs) else len(hrs) * k
                    # test_lables_count_std_num = sum(sum(test_lables))
                    HR = sum(hrs) / test_lables_count_std_num if test_lables_count_std_num != 0 else 0
                    MAP = sum(aps) / len(aps) if len(aps) != 0 else 0
                    NDCG = sum(ndcgs) / len(ndcgs) if len(ndcgs) != 0 else 0
                    logging.info(
                        '|Predict Type {}| Predict {} Hours | Epoch {:02d} / {:04d} | HR:{} | MAP:{} | NDCG:{}'.
                        format(now_crime_id, hour_list[tp], epoch + 1, exp_args.train_epoch, HR, MAP, NDCG))
                    # file = open('/home/hongyi/CrimePrediction_LAB_WIN/data/NYC_hotspots_{}.pkl'.format(now_crime_id), 'wb')
                    # pickle.dump(out_res0, file)
                    # pickle.dump(out_res1, file)
                    # pickle.dump(test_lables_count, file)
                    # file.close()
                    # print("Hotspots file dump.")

                    # mean_test_loss = np.mean(loss_item)
                    # r_precision, r_recall, r_f1 = get_score(aTP, aTN, aFP, aFN)
                    # logging.info(
                    #     '|Predict {} Hours| Epoch {:02d} / {:04d} | Epoch Loss :{:8f} | Test Mean Loss:{:.8f} | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f}'.
                    #     format(hour_list[tp], epoch + 1, exp_args.train_epoch,
                    #            epoch_loss[tp] / exp_args.batch_per_epoch,
                    #            mean_test_loss,
                    #            r_precision, r_recall, r_f1))
                    # logging.info(
                    #     '|Predict {} Hours| Epoch {:02d} / {:04d} | TP:{} | TN:{} | FP:{} | FN:{}'.
                    #     format(hour_list[tp], epoch + 1, exp_args.train_epoch, aTP, aTN, aFP, aFN))
                    #
                    # if r_f1 > res_f1[tp]:
                    #     res_precision[tp], res_recall[tp], res_f1[tp] = r_precision, r_recall, r_f1
                    #     res_TP[tp], res_TN[tp], res_FP[tp], res_FN[tp] = aTP, aTN, aFP, aFN
                    #
                    #     model_path = os.path.join(exp_args.trained_model,
                    #                               'trained_{}_fine_tuning_{}_f1-{:4f}.pth'.format(now_crime_id,
                    #                                                                               model_start_time,
                    #                                                                               res_f1[tp]
                    #                                                                               ))
                    #     torch.save(model, model_path)
                    #     # best_all_f = all_f
                    #     # torch.save(model, model_path)

        # for tp in range(interval_choose, interval_choose + 1):
        #     logging.info(
        #         'Final Result | Predict {} Hours | Precision:{:.4f} | Recall:{:.4f} | F1:{:.4f} | TP:{} | TN:{} | FP:{} | FN:{}'.
        #         format(hour_list[tp], res_precision[tp], res_recall[tp], res_f1[tp], res_TP[tp], res_TN[tp], res_FP[tp],
        #                res_FN[tp]))

        # save model_v3
        # model_path = os.path.join(exp_args.trained_model,
        #                           'trained_crime-{}_model_wd_{}.pth'.format(now_crime_id, model_start_time))
        # torch.save(model, model_path)

        # pass
