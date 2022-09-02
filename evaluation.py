import torch
import numpy as np


class Evaluation:
    """
    Handles evaluation on a given POI dataset and loader.
    The two metrics are MAP and recall@n.
    Our model predicts sequence of next locations determined by the sequence_length at one pass.
    During evaluation, we treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations, and we can compute the two metrics.

    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.

    Using the --report_user argument one can access the statistics per user.
    """

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting

    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)  # 初始化隐状态

        with torch.no_grad():
            iter_cnt = 0
            recall1 = 0
            recall5 = 0
            recall10 = 0
            average_precision = 0.

            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)
            reset_count = torch.zeros(self.user_count)

            for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(self.setting.device)  # 去掉dataloader中的长度为1的batch_size
                t = t.squeeze().to(self.setting.device)
                s = s.squeeze().to(self.setting.device)
                y = y.squeeze()
                y_t = y_t.squeeze().to(self.setting.device)
                y_s = y_s.squeeze().to(self.setting.device)

                active_users = active_users.to(self.setting.device)

                # evaluate:
                out, h = self.trainer.evaluate(x, t, s, y_t, y_s, h, active_users)

                for j in range(self.setting.batch_size):
                    # o contains a per-user list of votes for all locations for each sequence entry
                    o = out[j]  # 取出第j个用户的序列信息

                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    # top 10 elements  这个方法就是让倒数第10个数是第10大的元素，且后面的元素虽无序但都比他大
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:]

                    y_j = y[:, j]  # 当前用户的20个标签

                    for k in range(len(y_j)):
                        if reset_count[active_users[j]] > 1:
                            continue  # todo skip already evaluated users.

                        # resort indices for k:
                        ind_k = ind[k]  # 当前地点的预测标签，就是地点编号的数组
                        # 因为前面那个argPartition只是保证了这十个值是最大10个，但内部没有排序，因此在这里排序 sort top 10 elements descending
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]

                        r = torch.tensor(r)
                        t = y_j[k]  # 当前地点答案标签

                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]  # 标签对应的预测概率
                        upper = np.where(r_kj > t_val)[0]  # np.where返回的是个tuple，取第0项能获得具体数据，且数据为索引，这里返回比预测概率更高的值所在索引
                        precision = 1. / (1 + len(upper))  # todo 这个精度计算实际上应该是MRR

                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r[:1]  # 这种写法是计算答案标签是否在预测的前1/5/10个预测标签中，若是就+1
                        u_recall5[active_users[j]] += t in r[:5]
                        u_recall10[active_users[j]] += t in r[:10]
                        u_average_precision[active_users[j]] += precision

            formatter = "{0:.8f}"
            for j in range(self.user_count):  # 遍历所有用户，累计所有用户的recall值
                iter_cnt += u_iter_cnt[j]
                recall1 += u_recall1[j]
                recall5 += u_recall5[j]
                recall10 += u_recall10[j]
                average_precision += u_average_precision[j]

                if self.setting.report_user > 0 and (j + 1) % self.setting.report_user == 0:
                    print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1',
                          formatter.format(u_recall1[j] / u_iter_cnt[j]), 'MAP',
                          formatter.format(u_average_precision[j] / u_iter_cnt[j]), sep='\t')

            print('recall@1:', formatter.format(recall1 / iter_cnt))
            print('recall@5:', formatter.format(recall5 / iter_cnt))
            print('recall@10:', formatter.format(recall10 / iter_cnt))
            print('MAP', formatter.format(average_precision / iter_cnt))
            print('predictions:', iter_cnt)
