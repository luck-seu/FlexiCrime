# Copyright (c) Facebook, Inc. and its affiliates.

from threading import Thread
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
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import datasets
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
model_path = "/home/tangjun/CrimePrediction/neural_stpp/experiments/attncnf64-64-64_concat_swish_ot0.0001_l2attn_tol0.0001_neural32-32gru_softplus_ot0.0001_cond_sharehidden_lr0.001_gc0_bsz3000x1_wd1e-06_s0_20230413_174903/model.pth"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(args.port)
dist.init_process_group("nccl", rank=0, world_size=1, timeout=datetime.timedelta(minutes=30))


def get_index_of_first_element_greater_than_x(arr, x):
    for i in range(len(arr)):
        if arr[i] > x:
            return i - 1
    return -1


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
        return torch.tensor([0.0]), torch.tensor([24.0])
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
        return datasets.CrimeNYC(datatest, split=split)
    else:
        raise ValueError(f"Unknown data option {data}")


def validate(model, test_loader, datatest, t0, t1, device):
    # model = model.to(device)
    model.eval()
    intensity = []
    # datatest = np.load("data/crimeNYC0/crime_nyc_test_theft.npy", allow_pickle=True)
    baseDate = pd.Timestamp("2016-01-01T00:00:00")
    for i in range(len(datatest)):
        time = (datatest[i][0] - baseDate).total_seconds() / 3600
        datatest[i][0] = time - find_max(time, 24)
    with torch.no_grad():
        for batch in test_loader:
            print(batch[0].shape, batch[1].shape, batch[2].shape)
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
                intensity.append(math.exp(sprob[index]) * intensitytime[i, index - 1])
            print(space_loglik.shape, time_loglik.shape)

    # model.train()
    return intensity


# model = torch.load(model_path)
# device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
# print(device)

def Test(testlist):
    # model = torch.load(model_path)
    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location={'cuda:0':'cuda:1'})
    # print(device)
    # model.to(device)
    test_set = load_data(args.data, testlist, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn,
    )
    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.data))
    # model.to(device)
    intensity = validate(model, test_loader, testlist, t0, t1, device)
    print(intensity)


if __name__ == '__main__':
    # lt = np.array([[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                [pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    #                ])



    lt = np.array([[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
                   [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]
                   ])
    print(len(lt))
    st = time.time()
    Test(lt)
    ed = time.time()
    print("time: ", ed - st)

    # lt = np.array([[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]])
    # Test(lt)


    # st = time.time()
    # # lt = np.array([[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    # #                [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]])
    # # Test(lt)
    # for i in range(4):
    #     lt = np.array([[pd.Timestamp("2016-01-02T00:00:00"), 1, 2], [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
    #                    [pd.Timestamp("2016-02-24T00:00:00"), 4, 1], [pd.Timestamp("2016-03-09T00:00:00"), 2, 4]])
    #     t = Thread(target=Test, args=(lt,))
    #     t.start()
    # ed = time.time()
    # print("time: ", ed - st)