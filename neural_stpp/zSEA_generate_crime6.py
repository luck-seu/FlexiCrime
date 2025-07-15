# -*- coding: utf-8 -*-
# @File    : zSEA_generate_crime0.py
# @Author  : juntang
# @Time    : 2023/4/28 10:27
import pickle

# Copyright (c) Facebook, Inc. and its affiliates.

import psutil
import argparse
import itertools
import datetime
import math
import numpy as np
import pandas as pd
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
import datasets_rare as datasets
from models.temporal.neural import ACTFNS as TPP_ACTFNS
import toy_datasets
from viz_dataset import MAPS

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, choices=MAPS.keys(), default="earthquakes_jp")
parser.add_argument("--model", type=str, choices=["cond_gmm", "gmm", "cnf", "tvcnf", "jumpcnf", "attncnf"],
                    default="gmm")
parser.add_argument("--tpp", type=str, choices=["poisson", "hawkes", "correcting", "neural"], default="hawkes")
parser.add_argument("--actfn", type=str, default="swish")
parser.add_argument("--tpp_actfn", type=str, choices=TPP_ACTFNS.keys(), default="softplus")
parser.add_argument("--hdims", type=str, default="64-64-64")
parser.add_argument("--layer_type", type=str, choices=["concat", "concatsquash"], default="concat")
parser.add_argument("--tpp_hdims", type=str, default="32-32")
parser.add_argument("--tpp_nocond", action="store_false", dest='tpp_cond')
parser.add_argument("--tpp_style", type=str, choices=["split", "simple", "gru"], default="gru")
parser.add_argument("--no_share_hidden", action="store_false", dest='share_hidden')
parser.add_argument("--solve_reverse", action="store_true")
parser.add_argument("--naive_hutch", action="store_true")
parser.add_argument("--tol", type=float, default=1e-4)
parser.add_argument("--otreg_strength", type=float, default=1e-4)
parser.add_argument("--tpp_otreg_strength", type=float, default=1e-4)

parser.add_argument("--warmup_itrs", type=int, default=0)
parser.add_argument("--num_iterations", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--gradclip", type=float, default=0)
parser.add_argument("--max_events", type=int, default=3000)
parser.add_argument("--test_bsz", type=int, default=32)

parser.add_argument("--experiment_dir", type=str, default="experiments")
parser.add_argument("--experiment_id", type=str, default=None)
parser.add_argument("--ngpus", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--logfreq", type=int, default=10)
parser.add_argument("--testfreq", type=int, default=30)
parser.add_argument("--port", type=int, default=None)
args = parser.parse_args()

args.port = int(np.random.randint(10000, 20000))
args.data = "crimeseattle"
args.test_bsz = 32

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(args.port)
rank = 0
dist.init_process_group("nccl", rank=rank, world_size=1, timeout=datetime.timedelta(minutes=30))

model_path = "/home/tangjun/CrimePrediction_LAB_PC/neural_stpp/trained_model/SEA/model6-1.pth"
crime_base_dataset = np.load('/home/tangjun/CrimePrediction_LAB_PC/neural_stpp/data/crimeSEA6/crime_seattle_TRESPASS_day7-1.npz')

model = torch.load(model_path)
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

def find_max(x, y):
    return (x // y) * y


def cast(tensor, device):
    return tensor.float().to(device)


def get_t0_t1(data):
    if data == "citibike":
        return torch.tensor([0.0]), torch.tensor([24.0])
    elif data == "covid_nj_cases":
        return torch.tensor([0.0]), torch.tensor([7.0])
    elif data == "earthquakes_jp":
        return torch.tensor([0.0]), torch.tensor([30.0])
    elif data == "pinwheel":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "gmm":
        return torch.tensor([0.0]), torch.tensor([toy_datasets.END_TIME])
    elif data == "fmri":
        return torch.tensor([0.0]), torch.tensor([10.0])
    elif data == "crimeseattle":
        return torch.tensor([0.0]), torch.tensor([3.0])
    else:
        raise ValueError(f"Unknown dataset {data}")


def load_data(data, datatest, split="train"):
    if data == "citibike":
        return datasets.Citibike(split=split)
    elif data == "covid_nj_cases":
        return datasets.CovidNJ(split=split)
    elif data == "earthquakes_jp":
        return datasets.Earthquakes(split=split)
    elif data == "pinwheel":
        return toy_datasets.PinwheelHawkes(split=split)
    elif data == "gmm":
        return toy_datasets.GMMHawkes(split=split)
    elif data == "fmri":
        return datasets.BOLD5000(split=split)
    elif data == "crimeseattle":
        return datasets.CrimeSeattle(crime_base_dataset, datatest, split=split)
    else:
        raise ValueError(f"Unknown data option {data}")


def get_index(arr, x):
    if arr[0] > x:
        return 0
    for i in range(1, len(arr)):
        if arr[i] > x:
            return i - 1
    return len(arr) - 1


def get_index_of_last_element_smaller_than_x(arr, x):
    for i in range(0, len(arr)):
        if arr[i] == 0 and i != 0 and arr[i - 1] != 0:
            return i - 1
        if arr[i] > x:
            return i
    return len(arr) - 1


def validate(model, test_loader, datatest, t0, t1, device, task_id=-1):
    model.eval()
    intensity = []
    baseDate = pd.Timestamp("2020-01-01T00:00:00")
    for i in range(len(datatest)):
        time = (datatest[i][0] - baseDate).total_seconds() / 3600 / 24
        datatest[i][0] = time - find_max(time, 7)


    all_epoch = len(test_loader)
    iter_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            iter_idx += 1
            event_times, spatial_locations, input_mask = map(lambda x: cast(x, device), batch)
            N, T, D = spatial_locations.shape
            pad = torch.zeros((N, 1)).to(device)
            input_mask = torch.cat([input_mask, pad], dim=1)
            newevent_times = torch.zeros((N, T + 1)).to(device)
            newspatial_locations = torch.zeros((N, T + 1, D)).to(device)
            indexs = []

            for i in range(N):
                index = get_index_of_last_element_smaller_than_x(event_times[i].reshape(-1), datatest[i][0])
                indexs.append(index)

                new_time = np.array(datatest[i][0])
                new_loc = np.array([datatest[i][1] - 2.5, datatest[i][2] - 4.5])
                event_time = np.array(event_times[i].cpu())
                spatial_location = np.array(spatial_locations[i].cpu())
                event_time = np.insert(event_time, index, new_time)
                spatial_location = np.insert(spatial_location, (index) * D, new_loc)

                newevent_times[i] = torch.tensor(event_time).to(device)
                newspatial_locations[i] = torch.tensor(spatial_location.reshape(-1, D)).to(device)

            intensitytime, space_loglik, time_loglik = model(newevent_times, newspatial_locations, input_mask, t0, t1)
            for i in range(N):
                result = math.exp(space_loglik[i][indexs[i]]) * intensitytime[i, indexs[i]]
                intensity.append(result)
            print("task id:{}, iter_idx:{} / all_epoch:{}".format(task_id, iter_idx, all_epoch))
    return intensity

def divide_into_N(data, n):
    quotient = len(data) // n
    remainder = len(data) % n
    result = []
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + quotient + 1
        else:
            end = start + quotient
        result.append(data[start:end])
        start = end
    return result


def task(tasklist, date, task_id, crime_id):
    test_set = load_data(args.data, tasklist, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )

    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.data))
    intensity = validate(model, test_loader, tasklist, t0, t1, device, task_id=task_id)

    write_name = 'crime{}-query-taskid-{}-{}.pkl'.format(crime_id, task_id, date)
    with open(os.path.join('train_data/SEA/{}'.format(crime_id), date, write_name), 'wb') as f:
        pickle.dump(intensity, f)
    f.close()

    return intensity



def get_dataset(crime_id, processes_num=20):
    st = time.time()

    dataset_date = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
    write_dir = os.path.join('train_data/SEA/{}'.format(crime_id), dataset_date)
    os.makedirs(write_dir)

    with open('/home/tangjun/CrimePrediction_LAB_PC/data/Seattle/pp_data/point-process-query-{}-2020-2022.pkl'.format(crime_id), 'rb') as f:
        pp_query = pickle.load(f)
    f.close()
    pp_query = sum(pp_query, [])
    crime_test_len = math.floor(len(pp_query) / 8)
    pp_query_train = pp_query[:-crime_test_len]

    # 分割数据
    # test
    # pp_query_train = pp_query_train[:1000]
    pp_query_list = divide_into_N(pp_query_train, processes_num)
    # task(pp_query_list[0], dataset_date, 0)
    # 多进程造数据
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=processes_num) as pool:
        result = [pool.apply_async(task, (pp_query_list[i], dataset_date, i, crime_id, )) for i in range(len(pp_query_list))]
        output = [p.get() for p in result]
        # print(output)
    ed = time.time()

    print('Time:', (ed - st) / 60)
    pass


def test(testlist):
    model = torch.load(model_path)
    device = torch.device(f'cuda:{rank:d}' if torch.cuda.is_available() else 'cpu')
    test_set = load_data(args.data, testlist, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )
    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.data))

    intensity = validate(model, test_loader, testlist, t0, t1, device)
    # print(intensity)
    return intensity

def dataset_collection(crime_id, date, processes_num=20, to_file='', order=''):
    all_data = []
    for i in range(processes_num):
        data_name = '/home/tangjun/CrimePrediction_LAB_PC/neural_stpp/train_data/SEA/{}/{}/crime{}-query-taskid-{}-{}.pkl'.format(
            crime_id, date+order, crime_id, i, date)
        with open(data_name, 'rb') as f:
            pp_data = pickle.load(f)
        f.close()
        all_data.extend(pp_data)
        print(i + 1, '/', processes_num)

    # 一条查询对应4个代表点
    for i in range(len(all_data)):
        all_data[i] = all_data[i].to('cpu')
    if to_file == '':
        write_name = '/home/tangjun/CrimePrediction_LAB_PC/neural_stpp/train_data/SEA/{}/{}/all-crime{}-query-{}.pkl'.format(
            crime_id, date, crime_id, date)
        with open(write_name, 'wb') as f:
            pickle.dump(all_data, f)
        f.close()
    else:
        name = 'pp-labels-{}-2020-2022.pkl'.format(crime_id)
        write_name = os.path.join(to_file, name)
        with open(write_name, 'wb') as f:
            pickle.dump(all_data, f)
        f.close()

def one_test():
    base = [[pd.Timestamp("2016-01-02T00:23:00"), 1, 2], [pd.Timestamp("2016-01-08T06:00:23"), 3, 2],
            [pd.Timestamp("2016-02-24T12:11:00"), 4, 1], [pd.Timestamp("2016-03-09T20:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            ]

    all_one = []
    for i in range(2):
        all_one.extend(base)
    st = time.time()
    print('len:', len(all_one))
    out = test(np.array(all_one))
    print(out)
    ed = time.time()
    print('one:', ed - st)

if __name__ == "__main__":
    # get_dataset(6, 2)
    dataset_collection(6, '2023-05-09_20-56', 2,
                       to_file='/home/tangjun/CrimePrediction_LAB_PC/data/Seattle/pp_data/2023-5-10_12-00-o1', order='-o1')
    # one_test()


    # demo = [[pd.Timestamp("2020-09-04T12:00:00"), 5, 3], [pd.Timestamp("2020-09-04T12:00:00"), 3, 2],
    #         [pd.Timestamp("2020-09-04T12:00:00"), 2, 1], [pd.Timestamp("2021-06-09T18:00:00"), 2, 1],
    #         [pd.Timestamp("2020-01-02T00:00:00"), 4, 2], [pd.Timestamp("2020-01-08T00:00:00"), 3, 2],
    #         [pd.Timestamp("2020-04-21T00:00:00"), 2, 1], [pd.Timestamp("2021-06-09T18:00:00"), 2, 4]]
    # print(test(demo))