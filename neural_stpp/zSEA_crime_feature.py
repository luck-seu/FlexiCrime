import argparse
import itertools
import datetime
import math
import multiprocessing
import pickle
import line_profiler

import numpy as np
import pandas as pd
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import datasets_new as datasets
from iterators import EpochBatchIterator
from models.temporal.neural import ACTFNS as TPP_ACTFNS
import toy_datasets
from viz_dataset import MAPS

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
parser.add_argument("--data",
                    type=str,
                    choices=MAPS.keys(),
                    default="earthquakes_jp")
parser.add_argument(
    "--model",
    type=str,
    choices=["cond_gmm", "gmm", "cnf", "tvcnf", "jumpcnf", "attncnf"],
    default="gmm")
parser.add_argument("--tpp",
                    type=str,
                    choices=["poisson", "hawkes", "correcting", "neural"],
                    default="hawkes")
parser.add_argument("--actfn", type=str, default="swish")
parser.add_argument("--tpp_actfn",
                    type=str,
                    choices=TPP_ACTFNS.keys(),
                    default="softplus")
parser.add_argument("--hdims", type=str, default="64-64-64")
parser.add_argument("--layer_type",
                    type=str,
                    choices=["concat", "concatsquash"],
                    default="concat")
parser.add_argument("--tpp_hdims", type=str, default="32-32")
parser.add_argument("--tpp_nocond", action="store_false", dest='tpp_cond')
parser.add_argument("--tpp_style",
                    type=str,
                    choices=["split", "simple", "gru"],
                    default="gru")
parser.add_argument("--no_share_hidden",
                    action="store_false",
                    dest='share_hidden')
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
args.test_bsz = 16
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(args.port)
dist.init_process_group("nccl",
                        rank=0,
                        world_size=1,
                        timeout=datetime.timedelta(minutes=30))



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


def get_index_of_last_element_smaller_than_x(arr, x):
    for i in range(0, len(arr)):
        if arr[i] == 0 and i != 0 and arr[i - 1] != 0:
            return i - 1
        if arr[i] > x:
            return i
    return len(arr) - 1


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


def validate(model, test_loader, datatest, t0, t1, device, task_id=-1):
    model.eval()
    intensity = []
    baseDate = pd.Timestamp("2016-01-01T00:00:00")

    all_epoch = len(test_loader)
    iter_idx = 0
    batch_indexs = test_loader.batch_sampler
    # batch_nonduplicate_indexs = set()

    with torch.no_grad():
        for batch in test_loader:
            # event_times, spatial_locations, input_mask = map(lambda x: cast(x, device), batch)
            event_times, spatial_locations, input_mask, time_context, space_context = map(
                lambda x: cast(x, device), batch)
            N, T, D = spatial_locations.shape
            intensitytime, space_loglik, time_loglik = model(
                event_times, spatial_locations, input_mask, time_context,
                space_context, t0, t1)

            # for i in range(N):
            #     result = math.exp(
            #         space_loglik[i][indexs[i]]) * intensitytime[i, indexs[i]]
            #     intensity.append(result.cpu())
            current_batch_indexs = batch_indexs[iter_idx]
            for i in range(N):
                # if current_batch_indexs[i] in batch_nonduplicate_indexs:
                #     print(f"{current_batch_indexs[i]} duplicate!")
                # else:
                #     batch_nonduplicate_indexs.add(current_batch_indexs[i])
                result = math.exp(
                    space_loglik[i][test_loader.dataset.testdata_indexs[
                        current_batch_indexs[i]]]) * intensitytime[
                             i, test_loader.dataset.
                             testdata_indexs[current_batch_indexs[i]]]
                intensity.append(result.cpu())
            iter_idx += 1
            print("task id:{}, iter_idx:{} / batch_size:{} / all_epoch:{}".
                  format(task_id, iter_idx, N, all_epoch))

    return intensity


# model = torch.load(model_path)
# device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')


def Test(testlist):
    test_set = load_data(args.data, testlist, split="test")
    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=args.test_bsz,
    #     shuffle=False,
    #     collate_fn=datasets.spatiotemporal_events_collate_fn_test,
    # )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        collate_fn=datasets.spatiotemporal_events_collate_fn_test,
        shuffle=False,
        # batch_sampler=test_set.batch_by_size(args.max_events),
        batch_sampler=test_set.batch_by_size(test_set.max_events))

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


def task(tasklist, date, task_id, crime_id):
    test_set = load_data(args.data, tasklist, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_bsz,
        shuffle=False,
        collate_fn=datasets.spatiotemporal_events_collate_fn_test,
    )

    t0, t1 = map(lambda x: cast(x, device), get_t0_t1(args.data))
    intensity = validate(model,
                         test_loader,
                         tasklist,
                         t0,
                         t1,
                         device,
                         task_id=task_id)
    write_name = 'crime{}-query-taskid-{}-{}.pkl'.format(
        crime_id, task_id, date)
    with open(
            os.path.join('train_data/NYC/{}'.format(crime_id), date,
                         write_name), 'wb') as f:
        pickle.dump(intensity, f)
    f.close()

    return intensity


def get_dataset(crime_id, processes_num=20):
    st = time.time()
    dataset_date = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
    write_dir = os.path.join('train_data/NYC/{}'.format(crime_id),
                             dataset_date)
    os.makedirs(write_dir)

    with open(
            '/home/tangjun/CrimePrediction/data/NYC/pp_data/point-process-query-{}-2016-2018.pkl'
                    .format(crime_id), 'rb') as f:
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
        result = [
            pool.apply_async(task, (
                pp_query_list[i],
                dataset_date,
                i,
                crime_id,
            )) for i in range(len(pp_query_list))
        ]
        output = [p.get() for p in result]
        # print(output)
    ed = time.time()
    print('TIME:', (ed - st) / 60)
    pass


def dataset_collection(crime_id, date, processes_num=20, to_file='', order=''):
    all_data = []
    for i in range(processes_num):
        data_name = '/home/tangjun/CrimePrediction_LAB_PC/neural_stpp/train_data/NYC/{}/{}/crime{}-query-taskid-{}-{}.pkl'.format(
            crime_id, date + order, crime_id, i, date)

        with open(data_name, 'rb') as f:
            pp_data = pickle.load(f)
        f.close()
        all_data.extend(pp_data)
        print(i + 1, '/', processes_num)

    # 一条查询对应4个代表点
    for i in range(len(all_data)):
        all_data[i] = all_data[i].to('cpu')
    if to_file == '':
        write_name = '/home/tangjun/CrimePrediction_LAB_PC/neural_stpp/train_data/NYC/{}/{}/all-crime{}-query-{}.pkl'.format(
            crime_id, date, crime_id, date)
        with open(write_name, 'wb') as f:
            pickle.dump(all_data, f)
        f.close()
    else:
        name = 'pp-labels-{}-2016-2018.pkl'.format(crime_id)
        write_name = os.path.join(to_file, name)
        with open(write_name, 'wb') as f:
            pickle.dump(all_data, f)
        f.close()


def get_one(processes_num=20):
    # dataset_date = time.strftime('%Y-%m-%d_%H-%M', time.localtime(time.time()))
    dataset_date = '2023-04-19_22-26'
    with open(
            '/home/tangjun/CrimePrediction/data/NYC/pp_data/point-process-query-10-2016-2018.pkl',
            'rb') as f:
        pp_query = pickle.load(f)
    f.close()
    pp_query = sum(pp_query, [])
    crime_test_len = math.floor(len(pp_query) / 8)
    pp_query_train = pp_query[:-crime_test_len]
    pp_query_list = divide_into_N(pp_query_train, processes_num)  # 只造缺失的9
    # a = pp_query_list[9][32*104 :32*104 + 32]

    task(pp_query_list[9], dataset_date, 9)


def multi_test():
    base = [
        [pd.Timestamp("2016-01-02T00:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
        [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
        [pd.Timestamp("2016-01-02T00:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
        [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
        [pd.Timestamp("2016-01-02T00:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
        [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
        [pd.Timestamp("2016-01-02T00:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
        [pd.Timestamp("2016-01-02T0a0:00:00"), 1, 2],
        [pd.Timestamp("2016-01-08T00:00:00"), 3, 2],
        [pd.Timestamp("2016-02-24T00:00:00"), 4, 1],
        [pd.Timestamp("2016-03-09T00:00:00"), 2, 4],
    ]
    all_multi = []
    num = 2
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


def one_test():  # 最后一维度用来占位

    base = [

        [pd.Timestamp('2016-01-01 04:48:00'), 1, 2, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 1, 2, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 1, 2, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 1, 2, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 1, 3, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 1, 3, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 1, 3, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 1, 3, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 1, 4, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 1, 4, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 1, 4, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 1, 4, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 1, 5, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 1, 5, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 1, 5, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 1, 5, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 1, 6, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 1, 6, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 1, 6, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 1, 6, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 2, 0, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 2, 0, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 2, 0, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 2, 0, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 2, 1, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 2, 1, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 2, 1, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 2, 1, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 2, 2, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 2, 2, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 2, 2, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 2, 2, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 2, 3, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 2, 3, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 2, 3, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 2, 3, 0],
        [pd.Timestamp('2016-01-01 04:48:00'), 2, 4, 0],
        [pd.Timestamp('2016-01-01 09:36:00'), 2, 4, 0],
        [pd.Timestamp('2016-01-01 14:24:00'), 2, 4, 0],
        [pd.Timestamp('2016-01-01 19:12:00'), 2, 4, 0],

    ]

    # all_one = []
    # for i in range(2):
    #     all_one.extend(base)
    all_one = base
    st = time.time()
    print('len:', len(all_one))
    out = Test(np.array(all_one))
    print(out)
    ed = time.time()
    print('one:', ed - st)


def get_different_1day():
    # 现在对于同样的区域在一个时间区间取得的点过程都是一样的。因为都压缩到一天了，所以只需要将同一区域不同时间段的取出来即可。用字典来存？
    # 现在要做8小时、16小时、24小时的看看
    n_lat_len, n_lon_len = 10, 6
    # 构建查询数据集
    baseDate = pd.Timestamp("2020-01-01T00:00:00")
    # 先处理所有的查询时间点
    all_query_points = []

    # 4hours
    left_time = baseDate
    tq = []
    for i in range(6):  # 一天几个窗口
        right_time = left_time + datetime.timedelta(hours=4)
        n_q_points = pd.date_range(start=left_time,
                                   end=right_time,
                                   periods=4 + 2).tolist()[1:-1]
        left_time = right_time
        tq.extend(n_q_points)
    all_query_points.append(tq)

    # 6hours
    left_time = baseDate
    tq = []
    for i in range(4):
        right_time = left_time + datetime.timedelta(hours=6)
        n_q_points = pd.date_range(start=left_time,
                                   end=right_time,
                                   periods=4 + 2).tolist()[1:-1]
        left_time = right_time
        tq.extend(n_q_points)
    all_query_points.append(tq)

    # 8hours
    left_time = baseDate
    tq = []
    for i in range(3):
        right_time = left_time + datetime.timedelta(hours=8)
        n_q_points = pd.date_range(start=left_time,
                                   end=right_time,
                                   periods=4 + 2).tolist()[1:-1]
        left_time = right_time
        tq.extend(n_q_points)
    all_query_points.append(tq)

    # 12hours
    left_time = baseDate
    tq = []
    for i in range(2):
        right_time = left_time + datetime.timedelta(hours=12)
        n_q_points = pd.date_range(start=left_time,
                                   end=right_time,
                                   periods=4 + 2).tolist()[1:-1]
        left_time = right_time
        tq.extend(n_q_points)
    all_query_points.append(tq)

    # 24hours
    left_time = baseDate
    tq = []
    for i in range(1):
        right_time = left_time + datetime.timedelta(hours=24)
        n_q_points = pd.date_range(start=left_time,
                                   end=right_time,
                                   periods=4 + 2).tolist()[1:-1]
        left_time = right_time
        tq.extend(n_q_points)
    all_query_points.append(tq)

    hours_list = [4, 6, 8, 12, 24]
    for id in range(len(all_query_points)):
        n_q = all_query_points[id]
        pp_q = []
        for i in range(n_lat_len):
            for j in range(n_lon_len):
                for nt in n_q:
                    pp_q.append([nt, i, j, 0])  # 最后一维直接占位
        write_name = 'query-1day-{}hours.pkl'.format(hours_list[id])
        with open(os.path.join('./neural_stpp/data/SEA', write_name), 'wb') as f:
            pickle.dump(pp_q, f)
        f.close()
    pass


def get_1day_dataset(crime_id = -1):
    n_lat_len, n_lon_len = 10, 6
    hours_list = [6, 8, 12, 24]
    for k in range(len(hours_list)):
        city_data = {}  # 使用字典进行存放
        h = hours_list[k]
        data_name = 'query-1day-{}hours.pkl'.format(h)
        with open(os.path.join('./neural_stpp/data/SEA', data_name),
                  'rb') as f:
            pp_query = pickle.load(f)
        f.close()
        m_out = Test(np.array(pp_query))
        # with open(
        #         './neural_stpp/train_data/NYC/0/crime-data-raw-{}hours.pkl'.
        #         format(h), 'wb') as f:
        #     pickle.dump(m_out, f)
        # f.close()
        out = [float(t.item()) for t in m_out]

        id = 0
        for i in range(n_lon_len):
            for j in range(n_lat_len):
                a_t = 24 // h
                n_data = []  # 一个地点，一组查询
                for t in range(a_t):
                    nd = []
                    for x in range(4):
                        nd.append(out[id])
                        id += 1
                    n_data.append(nd)
                city_data[(i, j)] = n_data
                pass

        with open(
                './neural_stpp/train_data/SEA/{}/crime-pp-feature-{}hours.pkl'.
                        format(crime_id, h), 'wb') as f:
            pickle.dump(city_data, f)
        f.close()
        # st = time.time()
        # print('len:', len(all_one))
        # out = Test(np.array(all_one))
        # print(out)
        # ed = time.time()
        # print('one:', ed - st)

    pass



if __name__ == '__main__':
    # 先进行1day查询的构建，使用get_different_1day()
    # 需要更改更改
    # 本文件下model_path、crime_base_dataset
    # datasets_new.py 下的crime_context
    # 0 1 2 5 6 8
    crime_id = 8
    model_path = "/home/hongyi/neural_stpp-theft/crime_model/SEATTLE/model_{}.pth".format(crime_id)
    crime_base_dataset = np.load(
        '/home/hongyi/neural_stpp-theft/data/Seattle/pp_data/crime_seattle_{}_24hour_15min_20200101-20220101.npz'.format(crime_id)
    )
    model = torch.load(model_path)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    # get_different_1day() # 构建不同尺度1天的查询
    get_1day_dataset(crime_id)



    # -----------------------------------------------------------------------------------------------------------
    # print(pd.Timestamp("2016-01-01T00:10:00"))
    # one_test()
    # get_dataset(0, 5)
    # get_one()
    # dataset_collection(0, '2023-05-09_15-14', 5, to_file='/home/tangjun/CrimePrediction_LAB_PC/data/NYC/pp_data/2023-5-10_12-00-o1', order='-o1')
    # multi_test()
    # one_test()

    # data_name = '/home/tangjun/CrimePrediction_LAB_WIN/neural_stpp/train_data/SEA/0/crime-pp-feature-6hours.pkl'
    # with open(data_name, 'rb') as f:
    #     pp_data = pickle.load(f)
    # f.close()
