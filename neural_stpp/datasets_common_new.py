# # Copyright (c) Facebook, Inc. and its affiliates.
# import pickle
# import re
# import numpy as np
# import torch
# import pandas as pd
# import math
#
#
# class SpatioTemporalDataset(torch.utils.data.Dataset):
#
#     def __init__(self, train_set, test_set, train):
#         self.S_mean, self.S_std = self._standardize(train_set)
#
#         S_mean_ = torch.cat([torch.zeros(1, 1).to(self.S_mean), self.S_mean], dim=1)
#         S_std_ = torch.cat([torch.ones(1, 1).to(self.S_std), self.S_std], dim=1)
#         self.dataset = [(torch.tensor(seq) - S_mean_) / S_std_ for seq in (train_set if train else test_set)]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def _standardize(self, dataset):
#         dataset = [torch.tensor(seq) for seq in dataset]
#         full = torch.cat(dataset, dim=0)
#         S = full[:, 1:]
#         S_mean = S.mean(0, keepdims=True)
#         S_std = S.std(0, keepdims=True)
#         return S_mean, S_std
#
#     def unstandardize(self, spatial_locations):
#         return spatial_locations * self.S_std + self.S_mean
#
#     def ordered_indices(self):
#         lengths = np.array([seq.shape[0] for seq in self.dataset])
#         indices = np.argsort(lengths)
#         return indices, lengths[indices]
#
#     def batch_by_size(self, max_events):
#         try:
#             from data_utils_fast import batch_by_size_fast
#         except ImportError:
#             raise ImportError('Please run `python setup.py build_ext --inplace`')
#
#         indices, num_tokens = self.ordered_indices()
#
#         if not isinstance(indices, np.ndarray):
#             indices = np.fromiter(indices, dtype=np.int64, count=-1)
#         num_tokens_fn = lambda i: num_tokens[i]
#
#         return batch_by_size_fast(
#             indices, num_tokens_fn, max_tokens=max_events, max_sentences=-1, bsz_mult=1,
#         )
#
#     def __getitem__(self, index):
#         return self.dataset[index]
#
#
# class Citibike(SpatioTemporalDataset):
#     splits = {
#         "train": lambda f: bool(re.match(r"20190[4567]\d\d_\d\d\d", f)),
#         "val": lambda f: bool(re.match(r"201908\d\d_\d\d\d", f)) and int(
#             re.match(r"201908(\d\d)_\d\d\d", f).group(1)) <= 15,
#         "test": lambda f: bool(re.match(r"201908\d\d_\d\d\d", f)) and int(
#             re.match(r"201908(\d\d)_\d\d\d", f).group(1)) > 15,
#     }
#
#     def __init__(self, split="train"):
#         assert split in self.splits.keys()
#         self.split = split
#         dataset = np.load("data/citibike/citibike.npz")
#         train_set = [dataset[f] for f in dataset.files if self.splits["train"](f)]
#         split_set = [dataset[f] for f in dataset.files if self.splits[split](f)]
#         super().__init__(train_set, split_set, split == "train")
#
#     def extra_repr(self):
#         return f"Split: {self.split}"
#
#
# class CovidNJ(SpatioTemporalDataset):
#
#     def __init__(self, split="train"):
#         assert split in ["train", "val", "test"]
#         self.split = split
#         dataset = np.load("data/covid19/covid_nj_cases.npz")
#         dates = dict()
#         for f in dataset.files:
#             dates[f[:8]] = 1
#         dates = list(dates.keys())
#
#         # Reduce contamination between train/val/test splits.
#         exclude_from_train = (dates[::27] + dates[1::27] + dates[2::27]
#                               + dates[3::27] + dates[4::27] + dates[5::27]
#                               + dates[6::27] + dates[7::27])
#         val_dates = dates[2::27]
#         test_dates = dates[5::27]
#         train_dates = set(dates).difference(exclude_from_train)
#         date_splits = {"train": train_dates, "val": val_dates, "test": test_dates}
#         train_set = [dataset[f] for f in dataset.files if f[:8] in train_dates]
#         split_set = [dataset[f] for f in dataset.files if f[:8] in date_splits[split]]
#         super().__init__(train_set, split_set, split == "train")
#
#     def extra_repr(self):
#         return f"Split: {self.split}"
#
#
# class Earthquakes(SpatioTemporalDataset):
#
#     def __init__(self, split="train"):
#         assert split in ["train", "val", "test"]
#         self.split = split
#         dataset = np.load("data/earthquakes/earthquakes_jp.npz")
#         exclude_from_train = (dataset.files[::30] + dataset.files[1::30] + dataset.files[2::30] + dataset.files[3::30]
#                               + dataset.files[4::30] + dataset.files[5::30] + dataset.files[6::30] + dataset.files[
#                                                                                                      7::30]
#                               + dataset.files[8::30] + dataset.files[9::30] + dataset.files[10::30])
#         val_files = dataset.files[3::30]
#         test_files = dataset.files[:7]
#         train_files = set(dataset.files).difference(exclude_from_train)
#         file_splits = {"train": train_files, "val": val_files, "test": test_files}
#         train_set = [dataset[f] for f in train_files]
#         split_set = [dataset[f] for f in file_splits[split]]
#         super().__init__(train_set, split_set, split == "train")
#
#     def extra_repr(self):
#         return f"Split: {self.split}"
#
#
# class CrimeNYC(SpatioTemporalDataset):
#
#     def __init__(self, base_dataset, datatest, split="train"):
#         assert split in ["train", "val", "test"]
#         self.split = split
#         dataset = base_dataset
#         data2test = datatest
#         testCollection = []
#
#         # exclude_from_train = (dataset.files[::30] + dataset.files[1::30] + dataset.files[2::30] + dataset.files[3::30]
#         #                       + dataset.files[4::30] + dataset.files[5::30] + dataset.files[6::30] + dataset.files[7::30]
#         #                       + dataset.files[8::30] + dataset.files[9::30] + dataset.files[10::30])
#         exclude_from_train = (dataset.files[::24] + dataset.files[1::24] +
#                               dataset.files[2::24] + dataset.files[3::24] +
#                               dataset.files[4::24] + dataset.files[5::24] +
#                               dataset.files[6::24] + dataset.files[7::24])
#         val_files = dataset.files[1::24]
#
#         baseDate = pd.Timestamp("2016-01-01T00:00:00")
#         for i in range(len(data2test)):
#             # t = math.floor((data2test[i][0] - baseDate).days / 3)
#             t = math.floor(
#                 (data2test[i][0] - baseDate).total_seconds() % 86400 / 60 / 15)
#             testCollection.append(t)
#         test_files = np.array(dataset.files)
#         test_files = test_files[testCollection]
#         test_files = list(test_files)
#         train_files = set(dataset.files).difference(exclude_from_train)
#
#         file_splits = {
#             "train": train_files,
#             "val": val_files,
#             "test": test_files
#         }
#         train_set = [dataset[f] for f in train_files]
#         split_set = [dataset[f] for f in file_splits[split]]
#         super().__init__(train_set, split_set, split == "train")
#
#     def extra_repr(self):
#         return f"Split: {self.split}"
#
#
# class CrimeSeattle(SpatioTemporalDataset):
#
#     def __init__(self, base_dataset, datatest, split="train"):
#         assert split in ["train", "val", "test"]
#         self.split = split
#         dataset = base_dataset
#         # dataset = np.load("data/crime_seattle_LARCENY-THEFT_day3.npz")
#         data2test = datatest
#         testCollection = []
#
#         exclude_from_train = (dataset.files[::24] + dataset.files[1::24] +
#                               dataset.files[2::24] + dataset.files[3::24] +
#                               dataset.files[4::24] + dataset.files[5::24] +
#                               dataset.files[6::24] + dataset.files[7::24])
#         val_files = dataset.files[1::24]
#
#         baseDate = pd.Timestamp("2016-01-01T00:00:00")
#         for i in range(len(data2test)):
#             # t = math.floor((data2test[i][0] - baseDate).days / 3)
#             t = math.floor(
#                 (data2test[i][0] - baseDate).total_seconds() % 86400 / 60 / 15)
#             testCollection.append(t)
#         test_files = np.array(dataset.files)
#         test_files = test_files[testCollection]
#         test_files = list(test_files)
#         train_files = set(dataset.files).difference(exclude_from_train)
#
#         file_splits = {
#             "train": train_files,
#             "val": val_files,
#             "test": test_files
#         }
#         train_set = [dataset[f] for f in train_files]
#         split_set = [dataset[f] for f in file_splits[split]]
#         super().__init__(train_set, split_set, split == "train")
#
#     def extra_repr(self):
#         return f"Split: {self.split}"
#
#
# class BOLD5000(SpatioTemporalDataset):
#     splits = {
#         "train": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) < 8,
#         "val": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) == 8,
#         "test": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) > 8,
#     }
#
#     def __init__(self, split="train"):
#         assert split in self.splits.keys()
#         self.split = split
#         dataset = np.load("data/bold5000/bold5000.npz")
#         train_set = [dataset[f] for f in dataset.files if self.splits["train"](f)]
#         split_set = [dataset[f] for f in dataset.files if self.splits[split](f)]
#         super().__init__(train_set, split_set, split == "train")
#
#     def extra_repr(self):
#         return f"Split: {self.split}"
#
#
# def spatiotemporal_events_collate_fn(data):
#     """Input is a list of tensors with shape (T, 1 + D)
#         where T may be different for each tensor.
#
#     Returns:
#         event_times: (N, max_T)
#         spatial_locations: (N, max_T, D)
#         mask: (N, max_T)
#     """
#     if len(data) == 0:
#         # Dummy batch, sometimes this occurs when using multi-GPU.
#         return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
#     dim = data[0].shape[1]
#     lengths = [seq.shape[0] for seq in data]
#     max_len = max(lengths)
#     padded_seqs = [torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0) if s.shape[0] != max_len else s for s
#                    in data]
#     data = torch.stack(padded_seqs, dim=0)
#     event_times = data[:, :, 0]
#     spatial_locations = data[:, :, 1:]
#     mask = torch.stack([torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)], dim=0) for seq_len in lengths])
#
#     return event_times, spatial_locations, mask
#
#
# # 测试使用，加id占位但是不适用
# def spatiotemporal_events_collate_fn_test(data):
#     """Input is a list of tensors with shape (T, 1 + D)
#         where T may be different for each tensor.
#
#     Returns:
#         event_times: (N, max_T)
#         spatial_locations: (N, max_T, D)
#         mask: (N, max_T)
#     """
#     if len(data) == 0:
#         # Dummy batch, sometimes this occurs when using multi-GPU.
#         return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
#     dim = data[0].shape[1]
#     lengths = [seq.shape[0] for seq in data]
#     max_len = max(lengths)
#     padded_seqs = [
#         torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0)
#         if s.shape[0] != max_len else s for s in data
#     ]
#     data = torch.stack(padded_seqs, dim=0)
#     # event_times = data[:, :, 0]
#     # spatial_locations = data[:, :, 1:]
#     event_times = data[:, :, 1]
#     spatial_locations = data[:, :, 2:]
#     mask = torch.stack([
#         torch.cat([torch.ones(seq_len),
#                    torch.zeros(max_len - seq_len)], dim=0)
#         for seq_len in lengths
#     ])
#
#     # return event_times, spatial_locations, mask
#
#     # 使用crime-context
#     with open('/home/tangjun/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_data/case-feature-crime-0-2016-2018.pkl',
#               'rb') as file:  # TODO:我这边好像需要对应修改，这样写的话
#         context_file = pickle.load(file)
#     time_context = []
#     space_context = []
#     for d in data:
#         tc = []
#         sc = []
#         for row in d:
#             tc.append(context_file[0][int(row[0])])
#             sc.append(context_file[1][int(row[0])])
#             # tc.append(context_file[0][0])
#             # sc.append(context_file[1][0])
#         time_context.append(tc)
#         space_context.append(sc)
#
#     time_context = torch.tensor(time_context)
#     space_context = torch.tensor(space_context)
#     return event_times, spatial_locations, mask, time_context, space_context
#
#
# Copyright (c) Facebook, Inc. and its affiliates.
import pickle
import re
import numpy as np
import torch
import pandas as pd
import math


def find_max(x, y):
    return (x // y) * y


def get_index_of_last_element_smaller_than_x(arr, x):
    arr = arr[:, 1].flatten()
    for i in range(0, len(arr)):
        if arr[i] == 0 and i != 0 and arr[i - 1] != 0:
            return i - 1
        if arr[i] > x:
            return i
    return len(arr) - 1


class SpatioTemporalDataset(torch.utils.data.Dataset):

    def __init__(self, train_set, test_set, train):
        self.S_mean, self.S_std = self._standardize(train_set)

        S_mean_ = torch.cat([torch.zeros(1, 1).to(self.S_mean), self.S_mean],
                            dim=1)
        S_std_ = torch.cat([torch.ones(1, 1).to(self.S_std), self.S_std],
                           dim=1)
        self.dataset = [(torch.tensor(seq) - S_mean_) / S_std_
                        for seq in (train_set if train else test_set)]

    def __len__(self):
        return len(self.dataset)

    def _standardize(self, dataset):
        dataset = [torch.tensor(seq) for seq in dataset]
        full = torch.cat(dataset, dim=0)
        S = full[:, 1:]
        S_mean = S.mean(0, keepdims=True)
        S_std = S.std(0, keepdims=True)
        return S_mean, S_std

    def unstandardize(self, spatial_locations):
        return spatial_locations * self.S_std + self.S_mean

    def ordered_indices(self):
        lengths = np.array([seq.shape[0] for seq in self.dataset])
        indices = np.argsort(lengths)
        return indices, lengths[indices]

    def batch_by_size(self, max_events):
        try:
            from data_utils_fast import batch_by_size_fast
        except ImportError:
            raise ImportError(
                'Please run `python setup.py build_ext --inplace`')

        indices, num_tokens = self.ordered_indices()

        if not isinstance(indices, np.ndarray):
            indices = np.fromiter(indices, dtype=np.int64, count=-1)
        num_tokens_fn = lambda i: num_tokens[i]

        return batch_by_size_fast(
            indices,
            num_tokens_fn,
            max_tokens=max_events,
            max_sentences=-1,
            bsz_mult=1,
        )

    def __getitem__(self, index):
        return self.dataset[index]


class Citibike(SpatioTemporalDataset):
    splits = {
        "train":
        lambda f: bool(re.match(r"20190[4567]\d\d_\d\d\d", f)),
        "val":
        lambda f: bool(re.match(r"201908\d\d_\d\d\d", f)) and int(
            re.match(r"201908(\d\d)_\d\d\d", f).group(1)) <= 15,
        "test":
        lambda f: bool(re.match(r"201908\d\d_\d\d\d", f)) and int(
            re.match(r"201908(\d\d)_\d\d\d", f).group(1)) > 15,
    }

    def __init__(self, split="train"):
        assert split in self.splits.keys()
        self.split = split
        dataset = np.load("data/citibike/citibike.npz")
        train_set = [
            dataset[f] for f in dataset.files if self.splits["train"](f)
        ]
        split_set = [
            dataset[f] for f in dataset.files if self.splits[split](f)
        ]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class CovidNJ(SpatioTemporalDataset):

    def __init__(self, split="train"):
        assert split in ["train", "val", "test"]
        self.split = split
        dataset = np.load("data/covid19/covid_nj_cases.npz")
        dates = dict()
        for f in dataset.files:
            dates[f[:8]] = 1
        dates = list(dates.keys())

        # Reduce contamination between train/val/test splits.
        exclude_from_train = (dates[::27] + dates[1::27] + dates[2::27] +
                              dates[3::27] + dates[4::27] + dates[5::27] +
                              dates[6::27] + dates[7::27])
        val_dates = dates[2::27]
        test_dates = dates[5::27]
        train_dates = set(dates).difference(exclude_from_train)
        date_splits = {
            "train": train_dates,
            "val": val_dates,
            "test": test_dates
        }
        train_set = [dataset[f] for f in dataset.files if f[:8] in train_dates]
        split_set = [
            dataset[f] for f in dataset.files if f[:8] in date_splits[split]
        ]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class Earthquakes(SpatioTemporalDataset):

    def __init__(self, split="train"):
        assert split in ["train", "val", "test"]
        self.split = split
        dataset = np.load("data/earthquakes/earthquakes_jp.npz")
        exclude_from_train = (dataset.files[::30] + dataset.files[1::30] +
                              dataset.files[2::30] + dataset.files[3::30] +
                              dataset.files[4::30] + dataset.files[5::30] +
                              dataset.files[6::30] + dataset.files[7::30] +
                              dataset.files[8::30] + dataset.files[9::30] +
                              dataset.files[10::30])
        val_files = dataset.files[3::30]
        test_files = dataset.files[:7]
        train_files = set(dataset.files).difference(exclude_from_train)
        file_splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        train_set = [dataset[f] for f in train_files]
        split_set = [dataset[f] for f in file_splits[split]]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class CrimeNYC(SpatioTemporalDataset):

    def __init__(self, base_dataset, datatest, split="train"):
        assert split in ["train", "val", "test"]
        self.split = split
        self.testdata_indexs = []

        dataset = base_dataset
        self.max_events = datatest.size

        data2test = datatest
        testCollection = []

        # exclude_from_train = (dataset.files[::30] + dataset.files[1::30] + dataset.files[2::30] + dataset.files[3::30]
        #                       + dataset.files[4::30] + dataset.files[5::30] + dataset.files[6::30] + dataset.files[7::30]
        #                       + dataset.files[8::30] + dataset.files[9::30] + dataset.files[10::30])
        exclude_from_train = (dataset.files[::24] + dataset.files[1::24] +
                              dataset.files[2::24] + dataset.files[3::24] +
                              dataset.files[4::24] + dataset.files[5::24] +
                              dataset.files[6::24] + dataset.files[7::24])
        val_files = dataset.files[1::24]

        baseDate = pd.Timestamp("2016-01-01T00:00:00")
        for i in range(len(data2test)):
            # t = math.floor((data2test[i][0] - baseDate).days / 3)
            t = math.floor(
                (data2test[i][0] - baseDate).total_seconds() % 86400 / 60 / 15)
            testCollection.append(t)
        test_files = np.array(dataset.files)
        test_files = test_files[testCollection]
        test_files = list(test_files)
        train_files = set(dataset.files).difference(exclude_from_train)

        file_splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        train_set = [dataset[f] for f in train_files]
        split_set = [dataset[f] for f in file_splits[split]]

        # (yihong):把插入放在这里做
        if split == "test":
            for i in range(len(testCollection)):
                time = testCollection[i]
                data2test[i][0] = (time -
                                   find_max(time, 15)) / 15 * 3  # 这里还需要修改吗

            testdata_length = len(split_set)
            for i in range(testdata_length):
                index = get_index_of_last_element_smaller_than_x(
                    split_set[i], data2test[i][0])
                self.testdata_indexs.append(index)
                new_event = np.array([
                    index - 1 if index - 1 > 0 else index + 1, data2test[i][0],
                    data2test[i][1], data2test[i][2]
                ])
                split_set[i] = np.insert(split_set[i],
                                         index,
                                         new_event,
                                         axis=0)
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class CrimeSeattle(SpatioTemporalDataset):

    def __init__(self, base_dataset, datatest, split="train"):
        assert split in ["train", "val", "test"]
        self.split = split
        dataset = base_dataset
        # dataset = np.load("data/crime_seattle_LARCENY-THEFT_day3.npz")
        data2test = datatest
        testCollection = []

        exclude_from_train = (dataset.files[::24] + dataset.files[1::24] +
                              dataset.files[2::24] + dataset.files[3::24] +
                              dataset.files[4::24] + dataset.files[5::24] +
                              dataset.files[6::24] + dataset.files[7::24])
        val_files = dataset.files[1::24]

        baseDate = pd.Timestamp("2016-01-01T00:00:00")
        for i in range(len(data2test)):
            # t = math.floor((data2test[i][0] - baseDate).days / 3)
            t = math.floor(
                (data2test[i][0] - baseDate).total_seconds() % 86400 / 60 / 15)
            testCollection.append(t)
        test_files = np.array(dataset.files)
        test_files = test_files[testCollection]
        test_files = list(test_files)
        train_files = set(dataset.files).difference(exclude_from_train)

        file_splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        train_set = [dataset[f] for f in train_files]
        split_set = [dataset[f] for f in file_splits[split]]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


class BOLD5000(SpatioTemporalDataset):
    splits = {
        "train": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) < 8,
        "val": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) == 8,
        "test": lambda f: int(re.match(r"\d\d(\d\d)\d\d", f).group(1)) > 8,
    }

    def __init__(self, split="train"):
        assert split in self.splits.keys()
        self.split = split
        dataset = np.load("data/bold5000/bold5000.npz")
        train_set = [
            dataset[f] for f in dataset.files if self.splits["train"](f)
        ]
        split_set = [
            dataset[f] for f in dataset.files if self.splits[split](f)
        ]
        super().__init__(train_set, split_set, split == "train")

    def extra_repr(self):
        return f"Split: {self.split}"


def spatiotemporal_events_collate_fn(data):
    """Input is a list of tensors with shape (T, 1 + D)
        where T may be different for each tensor.

    Returns:
        event_times: (N, max_T)
        spatial_locations: (N, max_T, D)
        mask: (N, max_T)
    """
    if len(data) == 0:
        # Dummy batch, sometimes this occurs when using multi-GPU.
        return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
    dim = data[0].shape[1]
    lengths = [seq.shape[0] for seq in data]
    max_len = max(lengths)
    padded_seqs = [
        torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0)
        if s.shape[0] != max_len else s for s in data
    ]
    data = torch.stack(padded_seqs, dim=0)
    event_times = data[:, :, 0]
    spatial_locations = data[:, :, 1:]
    mask = torch.stack([
        torch.cat([torch.ones(seq_len),
                   torch.zeros(max_len - seq_len)], dim=0)
        for seq_len in lengths
    ])

    return event_times, spatial_locations, mask


# 测试使用，加id占位但是不适用
def spatiotemporal_events_collate_fn_test(data):
    """Input is a list of tensors with shape (T, 1 + D)
        where T may be different for each tensor.

    Returns:
        event_times: (N, max_T)
        spatial_locations: (N, max_T, D)
        mask: (N, max_T)
    """
    if len(data) == 0:
        # Dummy batch, sometimes this occurs when using multi-GPU.
        return torch.zeros(1, 1), torch.zeros(1, 1, 2), torch.zeros(1, 1)
    dim = data[0].shape[1]
    lengths = [seq.shape[0] for seq in data]
    max_len = max(lengths)
    padded_seqs = [
        torch.cat([s, torch.zeros(max_len - s.shape[0], dim).to(s)], 0)
        if s.shape[0] != max_len else s for s in data
    ]
    data = torch.stack(padded_seqs, dim=0)
    # event_times = data[:, :, 0]
    # spatial_locations = data[:, :, 1:]
    event_times = data[:, :, 1]
    spatial_locations = data[:, :, 2:]
    mask = torch.stack([
        torch.cat([torch.ones(seq_len),
                   torch.zeros(max_len - seq_len)], dim=0)
        for seq_len in lengths
    ])

    # return event_times, spatial_locations, mask

    # 使用crime-context
    with open(
            '/home/tangjun/CrimePrediction_LAB_WIN/experiment_v2/NYC/train_data/case-feature-crime-13-2016-2018.pkl',
            'rb') as file:
        context_file = pickle.load(file)
    time_context = []
    space_context = []
    for d in data:
        tc = []
        sc = []
        for row in d:
            tc.append(context_file[0][int(row[0])])
            sc.append(context_file[1][int(row[0])])
            # tc.append(context_file[0][0])
            # sc.append(context_file[1][0])
        time_context.append(tc)
        space_context.append(sc)

    time_context = torch.tensor(time_context)
    space_context = torch.tensor(space_context)
    return event_times, spatial_locations, mask, time_context, space_context
