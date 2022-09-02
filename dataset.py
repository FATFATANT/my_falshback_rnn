import random
from enum import Enum
import torch
from torch.utils.data import Dataset


class Split(Enum):
    """
    Defines whether to split for train or test.
    """
    TRAIN = 0
    TEST = 1


class Usage(Enum):
    """
    Each user has a different amount of sequences.
    The usage defines how many sequences are used:
        MIN: only as many as the minimal user has
        MAX: each sequence of any user is used (default)
        CUSTOM: up to a fixed amount if available.

    The unused sequences are discarded. This setting applies after the train/test split.
    """

    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2


class PoiDataset(Dataset):
    """
    Our Point-of-interest pytorch dataset: To maximize GPU workload we organize the data in batches of
    "user" x "a fixed length sequence of locations". The active users have at least one sequence in the batch.
    In order to fill the batch all the time we wrap around the available users:
    ** if an active user runs out of locations we replace him with a new one.
    When there are no unused users available we reuse already processed ones.
    This happens if a single user was way more active than the average user.
    The batch guarantees that each sequence of each user was processed at least once.  **

    This data management has the implication that ** some sequences might be processed twice (or more) per epoch **.
    During training, you should call PoiDataset::shuffle_users before the start of a new epoch.
    This leads to more stochastic as different sequences will be processed twice.
    During testing, you *have to* keep track of the already processed users.

    Working with a fixed sequence length omits awkward code by removing only few of the latest checkins per user.
    We work with an 80/20 train/test spilt, where test check-ins are strictly after training checkins.
    To obtain at least one test sequence with label we require any user to have at least
    (5*<sequence-length>+1) checkins in total. 这就是最短checks为101的由来
    """

    def reset(self):
        # reset training state:
        self.next_user_idx = 0  # current user index to add
        self.active_users = []  # current active users，初始化为200个1
        self.active_user_seq = []  # current active users sequences，初始化为200个0
        self.user_permutation = []  # shuffle users during training，初始化为

        # set active users:
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
            self.active_users.append(i)
            self.active_user_seq.append(0)

        # use 1:1 permutation:
        for i in range(len(self.users)):
            self.user_permutation.append(i)

    def shuffle_users(self):  # todo 似乎每个epoch是随机取个200个用户
        random.shuffle(self.user_permutation)  # 用户的序号数组，将其打乱
        # reset active users:
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.batch_size):
            self.next_user_idx = (self.next_user_idx + 1) % len(self.users)  # todo 这个取余没看懂，感觉不需要
            self.active_users.append(self.user_permutation[i])  # todo 费这个劲一个个加进去干啥？直接切片前200不行吗？
            self.active_user_seq.append(0)  # 一堆0

    def __init__(self, users, times, coords, locs, sequence_length, batch_size, split, usage, loc_count,
                 custom_seq_count):
        self.next_user_idx = None
        self.user_permutation = None
        self.active_user_seq = None
        self.active_users = None
        self.users = users
        self.times = times
        self.coords = coords
        self.locs = locs
        self.labels = []  # labels应是地点序号，lbl_times和lbl_coords应是对应的时间和坐标
        self.lbl_times = []
        self.lbl_coords = []
        self.sequences = []  # todo 关于sequences的这些个数组应该就是flashback的窗口大小裁剪后
        self.sequences_times = []
        self.sequences_coords = []
        self.sequences_labels = []
        self.sequences_lbl_times = []
        self.sequences_lbl_coords = []
        self.sequences_count = []
        self.Ps = []  # todo Ps和Qs也不知道是个啥
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.batch_size = batch_size  # 200，应该对应一次处理两百个用户的数据
        self.loc_count = loc_count  # 总共的地点数
        self.custom_seq_count = custom_seq_count  # todo 啥？

        self.reset()

        # collect locations:
        for i in range(loc_count):  # todo 不清楚干吗用
            self.Qs[i, 0] = i

            # align labels to locations (shift by one)
        for i, loc in enumerate(locs):  # 给每个用户组装标签
            self.locs[i] = loc[:-1]  # 错开一项，使得后一项是前一项的答案
            self.labels.append(loc[1:])
            # adapt time and coords:
            self.lbl_times.append(self.times[i][1:])  # 保存对应的时间和坐标
            self.lbl_coords.append(self.coords[i][1:])
            self.times[i] = self.times[i][:-1]
            self.coords[i] = self.coords[i][:-1]

        # split to training / test phase:  遍历的后三项分别为前三项的标签
        for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(
                zip(self.times, self.coords, self.locs, self.labels, self.lbl_times, self.lbl_coords)):
            train_thr = int(len(loc) * 0.8)  # 将轨迹集合中的每条轨迹进行裁剪，前80%作为训练集，后20%作为测试集
            if split == Split.TRAIN:
                self.times[i] = time[:train_thr]
                self.coords[i] = coord[:train_thr]
                self.locs[i] = loc[:train_thr]
                self.labels[i] = label[:train_thr]
                self.lbl_times[i] = lbl_time[:train_thr]
                self.lbl_coords[i] = lbl_coord[:train_thr]
            if split == Split.TEST:
                self.times[i] = time[train_thr:]
                self.coords[i] = coord[train_thr:]
                self.locs[i] = loc[train_thr:]
                self.labels[i] = label[train_thr:]
                self.lbl_times[i] = lbl_time[train_thr:]
                self.lbl_coords[i] = lbl_coord[train_thr:]

        # split location and labels to sequences:
        self.max_seq_count = 0  # min和max这样初始化设置是为了第一次比较时最大最小值都能被正常更新
        self.min_seq_count = 10000000
        self.capacity = 0
        for i, (time, coord, loc, label, lbl_time, lbl_coord) in enumerate(
                zip(self.times, self.coords, self.locs, self.labels, self.lbl_times, self.lbl_coords)):  # 此时是裁剪后的数据
            seq_count = len(loc) // sequence_length
            assert seq_count > 0, 'fix seq-length and min-checkins to have at least one test sequence in a 80/20 split!'
            seqs = []
            seq_times = []
            seq_coords = []
            seq_lbls = []
            seq_lbl_times = []
            seq_lbl_coords = []
            for j in range(seq_count):  # 将轨迹数据切分为若干个等长的序列
                start = j * sequence_length
                end = (j + 1) * sequence_length
                seqs.append(loc[start:end])
                seq_times.append(time[start:end])
                seq_coords.append(coord[start:end])
                seq_lbls.append(label[start:end])
                seq_lbl_times.append(lbl_time[start:end])
                seq_lbl_coords.append(lbl_coord[start:end])
            self.sequences.append(seqs)  # 保存序列数据
            self.sequences_times.append(seq_times)
            self.sequences_coords.append(seq_coords)
            self.sequences_labels.append(seq_lbls)
            self.sequences_lbl_times.append(seq_lbl_times)
            self.sequences_lbl_coords.append(seq_lbl_coords)
            self.sequences_count.append(seq_count)  # 每个用户的序列数
            self.capacity += seq_count  # 累加每个用户的序列数，69101
            self.max_seq_count = max(self.max_seq_count, seq_count)  # 前面初始化max为0是为了让实际的序列数肯定大于初始化值
            self.min_seq_count = min(self.min_seq_count, seq_count)  # min也是同样的道理

        # statistics
        if self.usage == Usage.MIN_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.MAX_SEQ_LENGTH:
            print(split, 'load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if self.usage == Usage.CUSTOM:
            print(split, 'load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:',
                  self.__len__())

    def sequences_by_user(self, idx):
        return self.sequences[idx]

    def __len__(self):
        """ Amount of available batches to process each sequence at least once.
        """

        if self.usage == Usage.MIN_SEQ_LENGTH:
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.batch_size)
        if self.usage == Usage.MAX_SEQ_LENGTH:
            # estimated capacity:
            estimated = self.capacity // self.batch_size
            return max(self.max_seq_count, estimated)
        if self.usage == Usage.CUSTOM:
            return self.custom_seq_count * (len(self.users) // self.batch_size)
        raise ValueError()

    def __getitem__(self, idx):
        """
        Against pytorch convention, we directly build a full batch inside __getitem__.
        Use a batch_size of 1 in your pytorch data loader.

        A batch consists of a list of active users,
        their next location sequence with timestamps and coordinates.

        y is the target location and y_t, y_s the targets timestamp and coordinates. Provided for
        possible use.

        reset_h is a flag which indicates when a new user has been replacing a previous user in the
        batch. You should reset these users hidden state to initial value h_0.
        """

        seqs = []
        times = []
        coords = []
        lbls = []
        lbl_times = []
        lbl_coords = []
        reset_h = []
        for i in range(self.batch_size):
            i_user = self.active_users[i]  # shuffle后的随机200个用户取出第i个
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]  # 该用户的总序列数
            if self.usage == Usage.MIN_SEQ_LENGTH:
                max_j = self.min_seq_count
            if self.usage == Usage.CUSTOM:
                max_j = min(max_j, self.custom_seq_count)  # use either the users maxima count or limit by custom count
            if j >= max_j:
                # replace this user in current sequence:
                i_user = self.user_permutation[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.user_permutation[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))  # 下面这些都是找出对应用户的第j条序列并保存
            times.append(torch.tensor(self.sequences_times[i_user][j]))
            coords.append(torch.tensor(self.sequences_coords[i_user][j]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            lbl_times.append(torch.tensor(self.sequences_lbl_times[i_user][j]))
            lbl_coords.append(torch.tensor(self.sequences_lbl_coords[i_user][j]))
            self.active_user_seq[i] += 1  # 这原本是个全0的数组，对应项+1变成1

        x = torch.stack(seqs, dim=1)  # 按这种叠法就是相当于每一列是一条序列，有200列
        t = torch.stack(times, dim=1)
        s = torch.stack(coords, dim=1)
        y = torch.stack(lbls, dim=1)
        y_t = torch.stack(lbl_times, dim=1)
        y_s = torch.stack(lbl_coords, dim=1)
        return x, t, s, y, y_t, y_s, reset_h, torch.tensor(self.active_users)
