import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """ Defines all settings in a single place using a command line interface.
    """

    def __init__(self):
        self.device = None
        self.report_user = None
        self.validate_epoch = None
        self.min_checkins = None
        self.batch_size = None
        self.sequence_length = None
        self.max_users = None
        self.dataset_file = None
        self.lambda_s = None
        self.lambda_t = None
        self.is_lstm = None
        self.rnn_factory = None
        self.epochs = None
        self.learning_rate = None
        self.weight_decay = None
        self.gpu = None
        self.hidden_dim = None
        self.guess_foursquare = None

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])  # 使用何种数据集是由命令行参数决定

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        '''###### settings ###### 从args中取出一系列参数存入setting实例中'''
        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()  # 在实例化RnnFactory传入的参数就决定了这个is_lstm的值
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.max_users = 0  # 0 = use all available users
        self.sequence_length = 20  # todo 不清楚这个sequence_length具体指啥
        self.batch_size = args.batch_size
        self.min_checkins = 101

        # evaluation        
        self.validate_epoch = args.validate_epoch  # 每5个epochs就会进行一轮验证
        self.report_user = args.report_user  # todo 不清楚这个参数有何用

        """### CUDA Setup ###"""
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    @staticmethod
    def parse_arguments(parser):
        # training
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')
        parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=100, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')

        # data management
        parser.add_argument('--dataset', default='checkins-gowalla.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')

        # evaluation
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')  # 每过5个epoch就验证一次
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')  # todo 这个值未知具体如何使用

    @staticmethod
    def parse_gowalla(parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=200, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')

    @staticmethod
    def parse_foursquare(parser):  # 这两个数据集的batch_size和空间数据decay不同
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=1024, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def __str__(self):
        return ('parse with foursquare default settings'
                if self.guess_foursquare else 'parse with gowalla default settings') + '\n' \
               + 'use device: {}'.format(self.device)
