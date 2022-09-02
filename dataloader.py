import os.path
import sys

from datetime import datetime

from dataset import PoiDataset, Usage


class PoiDataloader:
    """ Creates datasets from our prepared Gowalla/Foursquare data files.
    The file consist of one check-in per line in the following format (tab separated):

    <user-id> <timestamp> <latitude> <longitude> <location-id>

    Check-ins for the same user have to be on continuous lines.
    Ids for users and locations are recreated and continuous from 0.
    """

    def __init__(self, max_users=0, min_checkins=0):
        """
        max_users limits the amount of users to load.
        min_checkins discards users with less than this amount of checkins.
        """

        self.max_users = max_users
        self.min_checkins = min_checkins

        self.user2id = {}  # 对用户编号重新映射，键为用户编号，值为递增序号
        self.poi2id = {}  # 对地点编号重新映射，键为地点编号，值为递增序号

        self.users = []
        self.times = []
        self.coords = []
        self.locs = []

    def create_dataset(self, sequence_length, batch_size, split, usage=Usage.MAX_SEQ_LENGTH, custom_seq_count=1):
        return PoiDataset(self.users.copy(),
                          self.times.copy(),
                          self.coords.copy(),
                          self.locs.copy(),
                          sequence_length,
                          batch_size,
                          split,
                          usage,
                          len(self.poi2id),
                          custom_seq_count)

    def user_count(self):
        return len(self.users)

    def locations(self):
        return len(self.poi2id)

    def read(self, file):
        if not os.path.isfile(file):
            print('[Error]: Dataset not available: {}. Please follow instructions under ./data/README.md'.format(file))
            sys.exit(1)

        # collect all users with min checkins:
        self.read_users(file)
        # collect checkins for all collected users:
        self.read_pois(file)

    def read_users(self, file):  # 这种计法主要是因为数据本身就是按用户号连续的，因此一行行读计数即可
        f = open(file, 'r')
        lines = f.readlines()

        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:  # 当前行是新一个用户时，统计上一个用户的总访问次数，大于阈值则保存用户
                    self.user2id[prev_user] = len(self.user2id)  # 键为用户编号，值为递增序号
                # else:
                #    print('discard user {}: to few checkins ({})'.format(prev_user, visit_cnt))
                prev_user = user
                visit_cnt = 1  # 这个初始值是1是由于进到这个分支就必然已经有一条记录了
                # 记录了足够的用户数就终止，但由于max_users=0，and前的第一个条件永远为false，因此永远不会break
                if 0 < self.max_users <= len(self.user2id):
                    break  # restrict to max users

    def read_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()

        # store location ids
        user_time = []
        user_coord = []
        user_loc = []

        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)  # 得到用户编号对应的序号值
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:  # 访问记录小于阈值的用户不保存
                continue  # user is not of interest
            user = self.user2id.get(user)

            time = (datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ") - datetime(1970, 1,
                                                                                  1)).total_seconds()  # 转换为unix seconds
            lat = float(tokens[2])
            long = float(tokens[3])
            coord = (lat, long)

            location = int(tokens[4])  # location nr 地点编号
            if self.poi2id.get(location) is None:  # get-or-set locations
                self.poi2id[location] = len(self.poi2id)  # 键为地点编号，值为升序序号
            location = self.poi2id.get(location)  # 这里根据键名取值，使得location赋值记录的是映射后的排序而不是原地点编号

            if user == prev_user:
                # insert in front!  插值时从列表头部插入，这应该是由于原数据是按时间降序的，现在重新升序排列
                user_time.insert(0, time)
                user_coord.insert(0, coord)
                user_loc.insert(0, location)
            else:  # 遍历到新用户时，将上一个用户的记录再放入一个数组中
                self.users.append(prev_user)
                self.times.append(user_time)
                self.coords.append(user_coord)
                self.locs.append(user_loc)

                # 重新开始记录当前用户的相关信息
                prev_user = user
                user_time = [time]
                user_coord = [coord]
                user_loc = [location]

        # process also the latest user in the for loop  结束for循环时最后一个用户的数据并没有保存，因此在此处处理
        self.users.append(prev_user)
        self.times.append(user_time)
        self.coords.append(user_coord)
        self.locs.append(user_loc)
