#!/usr/bin/env python
import cv2
import os
import argparse
import json
import shutil

import numpy as np
import torch
import skvideo.io
#import matplotlib.pyplot as plt

from .io import IO
import tools
import tools.utils as utils
from numpy.lib.format import open_memmap
from PIL import Image
import pickle
import yaml



class Mydemo(IO):
    """
        Demo for SBU Skeleton-based Action Recgnition
    """
    
    def start(self):
        
        video_id = int(self.arg.video_id)
        wrong_label = self.arg.wrong_label
        wrong_result = self.arg.wrong_result
        config = yaml.load(open(self.arg.config, 'rb'))
        analyse_file = config['weights'].split('/')[-2]
        
        #print(set_num, video_id, wrong_label, wrong_result)
        
        def count_length(video):
            length = 0
            for i in range(len(video[0,:,0,0])):
                if video[0,i,0,0]:
                    length = i
                else:
                    pass
        
            return length

        #load data 
        file = '/home/a123/cjy/st-gcn-master/data/NTU_Interaction/NTU/xview/val_pixel.npy'
        label_file = '/home/a123/cjy/st-gcn-master/data/NTU_Interaction/xview/ntu_relation/val_label_2per.pkl'
        label_data = pickle.load(open(label_file, 'rb'))
        output_result_dir = self.arg.output_dir
        set_model = file.split('/')[-1].split('_')[0]
        
        #output for all demo
        output_result_path = output_result_dir + '{}_{}_{}.mp4'.format(set_model, video_id, label_data[0][video_id].split('.')[0])

        data_np = np.load(file)
 
        #demo data preparation
        #choose the test video 
        video_data = data_np[video_id]
        video_T = count_length(video_data)
        print('length of video:', video_T)
        data = video_data[:,0:video_T,:,:]
        data = torch.from_numpy(data) 
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()

        # extract feature
        print('\nNetwork forwad...')
        #self.model.eval()
        #output, feature, A_matrix = self.model.extract_feature(data)
        #output = output[0]
        #feature = feature[0]
        #A_matrix = abs(A_matrix)
        #A_weight = A_matrix.sum(axis=0)
        #intensity = (feature*feature).sum(dim=0)**0.5
        #intensity = intensity.cpu().detach().numpy()
      

        # visualization
        print('\nVisualization...')
        
        print('video label:', label_data[0][video_id], label_data[1][video_id])
 
        #visualization data prepartion
        pose = video_data
        #feature = intensity 
        edge = self.model.graph.edge 
        neighbor_link1 = self.model.graph.neighbor_1link
        neighbor_link2 = self.model.graph.neighbor_2link
        neighbor_link = neighbor_link1 + neighbor_link2
        relation_link = self.model.graph.relation_link
        height=1080 
        _, T, V, M = pose.shape
        print (pose.shape)

        
        #save root make
        #if not os.path.exists(output_result_dir):
                 #os.makedirs(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path, outputdict={'-b': '300000000'})
        
        #begin visual
        for t in range(video_T):
            frame = np.zeros((1080, 1920))
            
            # image resize
            #HH, WW = frame.shape
            #frame = cv2.resize(frame, (height * WW // HH //2 , height//2))
            #frame = cv2.resize(frame, (HH//2 , WW))
            H, W = frame.shape
            scale_factor = 2 * height / 1080  
             
            # draw skeleton
            skeleton = frame * 0 
            for m in range(M):
                '''
                for i, j in relation_link:
                    xi = int(pose[0, t, i, m] * W)
                    yi = int(pose[1, t, i, m] * H)
                    xj = int(pose[0, t, j, m] * W)
                    yj = int(pose[1, t, j, m] * H)
                    
                    #cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil(2 * scale_factor)))
                    #cv2.line(skeleton, (xi, yi), (xj, yj), (255, 0, 0), int(np.ceil(10 * A_weight[i,j] * scale_factor)))
                '''
                for i, j in neighbor_link1:
                    xi = int(pose[0, t, i, m])
                    yi = int(pose[1, t, i, m])
                    xj = int(pose[0, t, j, m])
                    yj = int(pose[1, t, j, m])
                    
                    #cv2.line(skeleton, (xi, yi), (xj, yj), (0, 0, 255), int(np.ceil(3 * scale_factor)))
                    cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil(3 * scale_factor)))
                for i, j in neighbor_link2:
                    xi = int(pose[0, t, i, m])
                    yi = int(pose[1, t, i, m])
                    xj = int(pose[0, t, j, m])
                    yj = int(pose[1, t, j, m])
                    
                    cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil(3 * scale_factor)))
                    #cv2.line(skeleton, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil(10 * A_weight[i,j] * scale_factor)))                

            img = skeleton

            # save video
            writer.writeFrame(img)
             
        print('visualization Done.')      
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))
        
        #SBU 15point visualization*****************************************************************    
        
        
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--output_dir',
            default='/home/a123/cjy/st-gcn-master/failure_case/',
            help='Path to save results')
        parser.add_argument('--height',
            default=1080,
            type=int,
            help='Path to save results')
        parser.add_argument('--video_id',
            default=0,
            help='video id in data path')
        parser.add_argument('--wrong_label',
            default=0,
            help='video label')
        parser.add_argument('--wrong_result',
            default=0,
            help='video predict result')
        parser.set_defaults(config='/home/a123/cjy/st-gcn-master/config/st_gcn/ntu-xview/demo_test.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
