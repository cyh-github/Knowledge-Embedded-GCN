#coding:utf-8
import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import pickle as pk
import skvideo.io
import yaml


import torchlight
from torchlight.torchlight.io import import_class
from torchlight.torchlight.io import str2bool
from torchlight.torchlight.io import DictAction
from torchlight.torchlight.io import IO
import tools.utils as utils


class Fusion():

    def __init__(self, argv=None):

        self.load_config(argv)
        self.load_data()
        self.load_model()
        self.joint_model = self.load_weights(self.joint_model, self.arg.joint_weights)
        self.part_model = self.load_weights(self.part_model, self.arg.part_weights)
        self.dev = self.arg.device


    def load_config(self, argv=None):
        parser = self.get_parser()
        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)
        self.arg = parser.parse_args(argv)


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = torch.utils.data.DataLoader(
                    dataset=Feeder(**self.arg.test_feeder_args),
                    batch_size=self.arg.test_batch_size,
                    shuffle=False,
                    num_workers=self.arg.num_worker * torchlight.torchlight.gpu.ngpu(
                        self.arg.device))

    def load_model(self):
        Joint_Model = import_class(self.arg.joint_model)
        self.joint_model = Joint_Model(**(self.arg.joint_model_args))

        Part_Model = import_class(self.arg.part_model)
        self.part_model = Part_Model(**(self.arg.part_model_args))

    def load_weights(self, model, weights_path):
        self.io = IO(self.arg.work_dir)

        return self.io.load_weights(model, weights_path)


    def start(self):

        #set model to eval mode
        self.joint_model.eval()
        self.part_model.eval()
        self.joint_model.to(self.dev)
        self.part_model.to(self.dev)

        fusion = []
        #set fusion ratio list to travil and find the max fusion result
        for k in list(np.arange(0.3,3,0.1).round(2)):
            joint_result_frag = []
            part_result_frag = []
            result_frag = []
            label_frag = []
            for joint_data, part_data, label in tqdm(self.data_loader):
                # get data
                joint_data = joint_data.float().to(self.dev)
                part_data = part_data.float().to(self.dev)
                label = label.long().to(self.dev)

                # inference
                with torch.no_grad():
                    joint_output = self.joint_model(joint_data)
                    part_output = self.part_model(part_data)
                    output = joint_output + float(k) * part_output

                joint_result_frag.append(joint_output.data.cpu().numpy())
                part_result_frag.append(part_output.data.cpu().numpy())
                result_frag.append(output.data.cpu().numpy())
                label_frag.append(label.data.cpu().numpy())

            self.joint_result = np.concatenate(joint_result_frag)
            self.part_result = np.concatenate(part_result_frag)
            self.result = np.concatenate(result_frag)
            self.label = np.concatenate(label_frag)

            joint_predict = self.joint_result.argsort()[:,-1]
            joint_correct = [i for i, l in enumerate(self.label) if joint_predict[i]==self.label[i]]
            joint_accuracy = len(joint_correct) * 1.0 / len(self.label)

            part_predict = self.part_result.argsort()[:, -1]
            part_correct = [i for i, l in enumerate(self.label) if part_predict[i] == self.label[i]]
            part_accuracy = len(part_correct) * 1.0 / len(self.label)

            predict = self.result.argsort()[:, -1]
            correct = [i for i, l in enumerate(self.label) if predict[i] == self.label[i]]
            accuracy = len(correct) * 1.0 / len(self.label)

            #print('1:{}  totally accuracy:'.format(k), accuracy)
            fusion.append(accuracy)

        print('joint accuracy:', joint_accuracy)
        print('part accuracy:', part_accuracy)
        print('fusion accuracy:', max(fusion))




    def test_single(self):
        '''
        函数功能：融合之前验证单路准确率
        '''

        #joint
        result_frag = []
        label_frag = []
        self.joint_model.eval()
        self.part_model.eval()
        self.joint_model.to(self.dev)
        self.part_model.to(self.dev)
        for joint_data, part_data, label in tqdm(self.data_loader):
            # get data
            joint_data = joint_data.float().to(self.dev)
            part_data = part_data.float().to(self.dev)
            label = label.long().to(self.dev)

            #inference
            with torch.no_grad():
                ########Select: single model banch to test########
                output = self.part_model(part_data)
                #output = self.joint_model(joint_data)

            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -1:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        print('Accuracy:', accuracy)

    


    @staticmethod
    def get_parser(add_help=False):
        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='IO Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default='/home/f1y/cyh/st-gcn-li/config/st_gcn/ntu-xsub/visual_org+3A.yaml', help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')


        # visulize and debug
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--joint_model', default=None, help='the model will be used')
        parser.add_argument('--joint_model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--joint_weights', default=None, help='the weights for network initialization')

        parser.add_argument('--part_model', default=None, help='the model will be used')
        parser.add_argument('--part_model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--part_weights', default=None, help='the weights for network initialization')

        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')

        #visualization add
        parser.add_argument('--video_id', type=int, default=1908, help='video choose to visualization')
        # endregion yapf: enable

        return parser

if __name__ == '__main__':
    fusion_work = Fusion()
    fusion_work.start()




