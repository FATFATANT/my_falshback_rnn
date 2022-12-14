import torch
from torch.utils.data import DataLoader
import numpy as np

from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from network import create_h0_strategy
from evaluation import Evaluation

'''
Main train script to invoke from commandline.
'''


def main():
    """### parse settings ###"""
    setting = Setting()
    setting.parse()  # 注意这里是setting自定义类中进行parser.parse()然后将args都赋值到这个实例的self中
    print(setting)

    """### load dataset ###"""
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)  # todo max_user默认是0，在作者代码逻辑中相当于无限大
    poi_loader.read(setting.dataset_file)
    dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # todo 这样的batch_size和shuffle参数可能是因为实际的功能是自己手动写的
    dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

    """### create flashback trainer ###"""
    trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
    h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
    trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory,
                    setting.device)
    evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)
    print('{} {}'.format(trainer, setting.rnn_factory))

    """###  training loop ###"""
    optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
    # epochs到达milestones时学习率乘上gamma进行学习率衰减
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

    for e in range(setting.epochs):
        h = h0_strategy.on_init(setting.batch_size, setting.device)  # (1, 200, 10)
        dataset.shuffle_users()  # shuffle users before each epoch!

        losses = []
        # i 一共344，即dataloader返回了345项，这长度来自于dataset的__len__
        for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader):
            # reset hidden states for newly added users
            for j, reset in enumerate(reset_h):  # reset_h是一个记录j是否为0的bool列表
                if reset:  # todo 将对应的序列的h还原到初始状态
                    if setting.is_lstm:
                        hc = h0_strategy.on_reset(active_users[0][j])
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
                    else:
                        h[0, j] = h0_strategy.on_reset(active_users[0][j])

            x = x.squeeze().to(setting.device)
            t = t.squeeze().to(setting.device)
            s = s.squeeze().to(setting.device)
            y = y.squeeze().to(setting.device)
            y_t = y_t.squeeze().to(setting.device)
            y_s = y_s.squeeze().to(setting.device)
            active_users = active_users.to(setting.device)
            optimizer.zero_grad()
            loss, h = trainer.loss(x, t, s, y, y_t, y_s, h, active_users)
            loss.backward(retain_graph=True)
            losses.append(loss.item())  # 将loss从tensor类型转化为python的float类型
            optimizer.step()

        # schedule learning rate:
        scheduler.step()

        # statistics:
        if (e + 1) % 1 == 0:  # todo 所有数%1肯定为0，这里这样写其实没有意义
            epoch_loss = np.mean(losses)
            print(f'Epoch: {e + 1}/{setting.epochs}')
            # print(f'Used learning rate: {scheduler.get_lr()[0]}')
            print(f'Used learning rate: {scheduler.get_last_lr()[0]}')
            print(f'Avg Loss: {epoch_loss}')
        if (e + 1) % setting.validate_epoch == 0:
            print(f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
            evaluation_test.evaluate()


if __name__ == '__main__':
    main()
