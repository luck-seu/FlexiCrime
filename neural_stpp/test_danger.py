# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import itertools
import datetime
import math
import multiprocessing
import pickle

import numpy as np
import pandas as pd
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import datasets_danger as datasets
from iterators import EpochBatchIterator
from models import CombinedSpatiotemporalModel, JumpCNFSpatiotemporalModel, SelfAttentiveCNFSpatiotemporalModel, \
    JumpGMMSpatiotemporalModel
from models.spatial import GaussianMixtureSpatialModel, IndependentCNF, JumpCNF, SelfAttentiveCNF
from models.spatial.cnf import TimeVariableCNF
from models.temporal import HomogeneousPoissonPointProcess, HawkesPointProcess, SelfCorrectingPointProcess, \
    NeuralPointProcess
from models.temporal.neural import ACTFNS as TPP_ACTFNS
from models.temporal.neural import TimeVariableODE
import toy_datasets
import utils
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
args.data = "crimenyc"
args.test_bsz = 32
model_path = "/home/tangjun/CrimePrediction_LAB_PC/neural_stpp/experiments/attncnf64-64-64_concat_swish_ot0.0001_l2attn_tol0.0001_neural32-32gru_softplus_ot0.0001_cond_sharehidden_lr0.001_gc0_bsz3000x1_wd1e-06_s0_20230413_174903/model.pth"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(args.port)
dist.init_process_group("nccl", rank=0, world_size=1, timeout=datetime.timedelta(minutes=30))


def get_index_of_first_element_greater_than_x(arr, x):
    index = -1
    for i in range(len(arr)):
        if arr[i] > x:
            index = i - 1
            break
    if index == -1:
        return 0
    else:
        return index


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
    elif data == "crimenyc":
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
    elif data == "crimenyc":
        return datasets.CrimeNYCTest(datatest, split=split)
    else:
        raise ValueError(f"Unknown data option {data}")


def validate(model, test_loader, datatest, t0, t1, device, task_id=-1):
    model.eval()
    intensity = []
    # datatest = np.load("data/crimeNYC0/crime_nyc_test_theft.npy", allow_pickle=True)
    baseDate = pd.Timestamp("2016-01-01T00:00:00")
    for i in range(len(datatest)):
        time = (datatest[i][0] - baseDate).total_seconds() / 3600 / 24
        datatest[i][0] = time - find_max(time, 4)

    all_epoch = len(test_loader)
    iter_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            iter_idx += 1
            # print(batch[0].shape, batch[1].shape, batch[2].shape)
            event_times, spatial_locations, input_mask = map(lambda x: cast(x, device), batch)
            N, T, D = spatial_locations.shape
            intensitytime, space_loglik, time_loglik = model(event_times, spatial_locations, input_mask, t0, t1)

            for i in range(N):
                index = get_index_of_first_element_greater_than_x(event_times[i].reshape(-1), datatest[i][0])
                lamdaprob = model.module.spatial_conditional_logprob_fn(datatest[i][0],
                                                                        event_times[i][:index].reshape(-1),
                                                                        spatial_locations[i][:index].reshape(-1, D), t0,
                                                                        t1)
                x = np.array([datatest[i][1], datatest[i][2]])
                sprob = lamdaprob(torch.from_numpy(x.reshape(1, D)).to(device))
                intensity.append((math.exp(sprob[index]) * intensitytime[i, index - 1]).to('cpu'))

            print("task id:{}, iter_idx:{} / all_epoch:{}".format(task_id, iter_idx, all_epoch))
    return intensity


model = torch.load(model_path)
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

def Test(testlist):
    test_set = load_data(args.data, testlist, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )
    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.data))
    intensity = validate(model, test_loader, testlist, t0, t1, device)
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


def task(tasklist, date, task_id):
    test_set = load_data(args.data, tasklist, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )

    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.data))
    intensity = validate(model, test_loader, tasklist, t0, t1, device, task_id=task_id)

    write_name = 'danger-query-taskid-{}-{}.pkl'.format(task_id, date)
    with open(os.path.join('train_data/10', date, write_name), 'wb') as f:
        pickle.dump(intensity, f)
    f.close()

    return intensity



def get_dataset(processes_num=20):
    dataset_date = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
    write_dir = os.path.join('train_data/10', dataset_date)
    os.makedirs(write_dir)

    with open('/home/tangjun/CrimePrediction/data/NYC/pp_data/point-process-query-10-2016-2018.pkl', 'rb') as f:
        pp_query = pickle.load(f)
    f.close()
    pp_query = sum(pp_query, [])
    crime_test_len = math.floor(len(pp_query) / 8)
    pp_query_train = pp_query[:-crime_test_len]

    # pp_query_train = [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                   ]

    # 分割数据

    pp_query_list = divide_into_N(pp_query_train, processes_num)
    # task(pp_query_list[0], dataset_date, 0)
    # 多进程造数据
    multiprocessing.set_start_method('spawn')
    with multiprocessing.Pool(processes=processes_num) as pool:
        result = [pool.apply_async(task, (pp_query_list[i], dataset_date, i,)) for i in range(len(pp_query_list))]
        output = [p.get() for p in result]
        # print(output)
    pass


def get_one(processes_num=20):
    # dataset_date = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
    dataset_date = '2023-04-19_22-26'
    with open('/home/tangjun/CrimePrediction/data/NYC/pp_data/point-process-query-10-2016-2018.pkl', 'rb') as f:
        pp_query = pickle.load(f)
    f.close()
    pp_query = sum(pp_query, [])
    crime_test_len = math.floor(len(pp_query) / 8)
    pp_query_train = pp_query[:-crime_test_len]
    pp_query_list = divide_into_N(pp_query_train, processes_num) # 只造缺失的9
    # a = pp_query_list[9][32*104 :32*104 + 32]

    task(pp_query_list[9], dataset_date, 9)

def multi_test():
    base = [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
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
            [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            ]
    all_multi = []
    num = 20
    for i in range(num):
        all_multi.append(base)
    multiprocessing.set_start_method('spawn')

    st = time.time()
    with multiprocessing.Pool(processes=num) as pool:
        result = [pool.apply_async(Test, (item,)) for item in all_multi]
        output = [p.get() for p in result]
        print(output)
    ed = time.time()
    print('multi:', ed - st)

    pass


def one_test():
    base = [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
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
            [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
            [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
            ]

    all_one = []
    for i in range(2):
        all_one.extend(base)
    st = time.time()
    tp = [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
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
          [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
          [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
          [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
          [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
          [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
          [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
          [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
          [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
          ]
    if (tp == all_one):
        print("1")
    out = Test(np.array(all_one))
    print(out)
    ed = time.time()
    print('one:', ed - st)


if __name__ == '__main__':
    # get_dataset()
    get_one()


    # multi_test()
    # one_test()

    # multiprocessing.set_start_method('spawn')
    #
    # st = time.time()
    #
    # # lt = np.array([[[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                ],
    # #               [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                ],
    # #               [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ],
    # #                [[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    # #                 [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                 [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
    # #                 ]
    # #                ]
    # #               )
    # lt = np.array([[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #
    #                ])
    #
    # # with multiprocessing.Pool(processes=12) as pool:
    # #     result = [pool.apply_async(Test, ([item],)) for item in lt]
    # #     output = [p.get() for p in result]
    # #     print(output)
    #
    # Test(lt)
    # ed = time.time()
    # print(ed - st)

    # st = time.time()
    # lt = np.array([[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                ])
    # Test(lt)
    # ed = time.time()
    # print(ed - st)

    # get_dataset()
