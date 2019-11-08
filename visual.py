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

SBU_label_dict = {'0': 'Approaching', '1': 'Leaving', '2': 'Kicking', '3': 'Punching',
              '4': 'Pushing', '5': 'Hugging', '6': 'ShakingHand', '7': 'Delivering'}

NTU_label_dict = {'0': 'punch/slap', '1': 'kicking', '2': 'pushing', '3': 'pat on back',
              '4': 'point finger', '5': 'hugging', '6': 'giving object', '7': 'touch pocket',
                  '8': 'shaking hands', '9': 'walking towards', '10': 'walking apart'}

def count_length(video):
    length = 0
    for i in range(len(video[0, :, 0, 0])):
        if video[0, i, 0, 0]:
            length = i
        else:
            pass

    return length


class Visual():

    def __init__(self, argv=None):

        self.load_config(argv)
        self.load_data()
        self.load_model()
        self.joint_model = self.load_weights(self.joint_model, self.arg.joint_weights)
        self.part_model = self.load_weights(self.part_model, self.arg.part_weights)
        self.org_model = self.load_weights(self.org_model, self.arg.org_weights)
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

        Org_Model = import_class(self.arg.org_model)
        self.org_model = Org_Model(**(self.arg.org_model_args))

    def load_weights(self, model, weights_path):
        self.io = IO(self.arg.work_dir)
        return self.io.load_weights(model, weights_path)



    def put_text(self, img, text, position, scale_factor=1):
        t_w, t_h = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
        H, W, _ = img.shape
        position = (int(W * position[1] - t_w * 0.5), int(H * position[0] - t_h * 0.5))
        params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
                  (0, 255, 0)) #RGB
        cv2.putText(img, text, *params)



    def visual(self):

        dataset = 'NTU' # SBU / NTU

        #load data
        joint_path = self.arg.test_feeder_args['joint_data_path']
        part_path = self.arg.test_feeder_args['part_data_path']
        pixel_path = './data/NTU_2p_pixel/{}/val_pixel.npy'.format(joint_path.split('/')[-2])
        label_path = self.arg.test_feeder_args['label_path']
        video_id = int(self.arg.video_id)

        self.joint_model.eval()
        self.part_model.eval()
        self.joint_model.to(self.dev)
        self.part_model.to(self.dev)

        test_data = np.load(joint_path)[video_id]   #############################Change for joint/part data
        label_data = pk.load(open(label_path, 'rb'))
        video_name = label_data[0][video_id]
        label = label_data[1][video_id]
        video_T = count_length(test_data)
        C,T,V,M = test_data.shape
        output_result_path = './visualization/{}/3A_{}_{}.mp4'.format(dataset, video_id, video_name.split('.')[0])

        data = torch.from_numpy(test_data)#test_data[:,0:video_T,:,:]
        data = data.unsqueeze(0)
        data = data.float().to(self.dev)

        # extract feature
        print('\nVisualization...', video_name, video_T)
        output, feature, Self_A = self.joint_model.extract_feature(data)
        output = output[0]
        feature = feature[0]  # feature [C,T,V,M] -> [128,T,30,1]
        data_A = Self_A['layer9'][0]

        # feature heat map
        intensity = (feature * feature).sum(dim=0) ** 0.5  # [T,V,M]
        intensity = intensity / (intensity.max() - intensity.min()) #normalization

        #get different 3A edges
        body_edge = self.joint_model.graph.edge
        k_edge = self.joint_model.k_graph.edge
        data_edge0 = [(i, j) for i in range(data_A.shape[-2]) for j in range(data_A.shape[-1]) if data_A[0, i, j] != 0]
        data_edge1 = [(i, j) for i in range(data_A.shape[-2]) for j in range(data_A.shape[-1]) if data_A[1, i, j] != 0]
        data_edge2 = [(i, j) for i in range(data_A.shape[-2]) for j in range(data_A.shape[-1]) if data_A[2, i, j] != 0]
        data_edge = data_edge0 # + data_edge1 + data_edge2
        #print('******learned dege:', data_edge)

        pred = output.argmax()
        print('\nNet prediction:', pred)

        if dataset == 'NTU':
            test_data = np.load(pixel_path)[video_id]
            pose = test_data[:,0:video_T,:,:]
            print('\nLabel:', label, NTU_label_dict['{}'.format(label)])
        else:
            pose = test_data[:,0:video_T,:,:]
            print('\nLabel:', label, SBU_label_dict['{}'.format(label)])

        writer = skvideo.io.FFmpegWriter(output_result_path)# outputdict={'-b': '300000000'}

        for t in range(video_T):
            frame =  np.full((1080, 1920, 3), 255, np.uint8)
            #frame = np.zeros((500, 500, 3), np.uint8)

            # image resize
            # HH, WW = frame.shape
            # frame = cv2.resize(frame, (height * WW // HH //2 , height//2))
            # frame = cv2.resize(frame, (HH//2 , WW))
            H, W, c = frame.shape
            scale_factor = 1

            # draw skeleton
            feature_heat = frame
            edge_map = frame
            for m in range(M):
                if dataset == 'NTU':

                    '''
                    #visual given edge 
                    for i, j in k_edge:
                        xi = int(pose[0, t, i, m])
                        yi = int(pose[1, t, i, m])
                        xj = int(pose[0, t, j, m])
                        yj = int(pose[1, t, j, m])
    
                        cv2.line(edge_map, (xi, yi), (xj, yj), (0, 0, 255), int(np.ceil(1 * scale_factor)))
                    
    
                    for i, j in data_edge:
                        xi = int(pose[0, t, i, m])
                        yi = int(pose[1, t, i, m])
                        xj = int(pose[0, t, j, m])
                        yj = int(pose[1, t, j, m])
    
                        cv2.line(feature_heat, (xi, yi), (xj, yj), (0, 255, 0), int(np.ceil(1 * scale_factor)))
                    '''

                    #visual natural edge in body
                    for i, j in body_edge:
                        xi = int(pose[0, t, i, m])
                        yi = int(pose[1, t, i, m])
                        xj = int(pose[0, t, j, m])
                        yj = int(pose[1, t, j, m])
                        if m == 0:
                            cv2.line(edge_map, (xi, yi), (xj, yj), (0, 0, 255), int(np.ceil(2 * scale_factor)))
                        else:
                            cv2.line(edge_map, (xi, yi), (xj, yj), (0, 255, 0), int(np.ceil(2 * scale_factor)))

                    #feature map heat
                    f = (intensity[t // 4, :, m] ** 5).cpu().detach().numpy()  # intensity [T,V,M]
                    if f.mean() != 0:
                        f = f / f.mean()
                    for v in range(V):
                        x = int(pose[0, t, v, m])
                        y = int(pose[1, t, v, m])
                        # print('feature map', f)
                        cv2.circle(feature_heat, (x, y), 0, (255, 0, 0),
                                   int(np.ceil(f[v] * 15 * scale_factor)))
                   

                    self.put_text(feature_heat, 'predction: ' + NTU_label_dict['{}'.format(pred)], (0.05, 0.2))  # position ratio (H,W)
                    self.put_text(feature_heat, 'label:' + NTU_label_dict['{}'.format(label)], (0.05, 0.6))


                # for SBU dataset
                else:
                    for i, j in body_edge:
                        xi = int(pose[0, t, i, m] * W)
                        yi = int(pose[1, t, i, m] * H)
                        xj = int(pose[0, t, j, m] * W)
                        yj = int(pose[1, t, j, m] * H)
                        cv2.line(edge_map, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil(2 * scale_factor)))

                    for i, j in k_edge:
                        xi = int(pose[0, t, i, m] * W)
                        yi = int(pose[1, t, i, m] * H)
                        xj = int(pose[0, t, j, m] * W)
                        yj = int(pose[1, t, j, m] * H)

                        cv2.line(edge_map, (xi, yi), (xj, yj), (0, 0, 255), int(np.ceil(1 * scale_factor)))

                    '''
                    #visual learned edge
                    for i, j in data_edge:
                        xi = int(pose[0, t, i, m] * W)
                        yi = int(pose[1, t, i, m] * H)
                        xj = int(pose[0, t, j, m] * W)
                        yj = int(pose[1, t, j, m] * H)
                        cv2.line(feature_heat, (xi, yi), (xj, yj), (0, 255, 0), int(np.ceil(1 * scale_factor)))
                    '''

                    f = (intensity[t // 4, :, m]**5).cpu().detach().numpy()  # intensity [T,V,M]
                    if f.mean() != 0:
                        f = f / f.mean()
                    for v in range(V):
                        x = int(pose[0, t, v, m] * W)
                        y = int(pose[1, t, v, m] * H)
                        cv2.circle(feature_heat, (x, y), 0, (255, 0, 0),
                                   int(np.ceil(f[v] * 20 * scale_factor)))

                    self.put_text(feature_heat, 'predction: ' + SBU_label_dict['{}'.format(label)], (0.05, 0.2))  # position ratio (H,W)
                    self.put_text(feature_heat, 'label:' + SBU_label_dict['{}'.format(label)], (0.05, 0.6))

            img = feature_heat + edge_map
            # save video
            writer.writeFrame(img)

        print('visualization Done.')
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))


    def visual_ND(self):

        dataset = 'NTU' # SBU / NTU

        #joint
        part_path = self.arg.test_feeder_args['part_data_path']
        pixel_path = './data/NTU_2p_pixel/{}/val_pixel.npy'.format(part_path.split('/')[-2])
        label_path = self.arg.test_feeder_args['label_path']
        video_id = int(self.arg.video_id)

        self.part_model.eval()
        self.part_model.to(self.dev)

        test_data = np.load(part_path)[video_id] # [C, T, V, M] -> [3,T,25,2]
        print(test_data.shape)
        label_data = pk.load(open(label_path, 'rb'))
        video_name = label_data[0][video_id]
        label = label_data[1][video_id]
        print(label_data[1])
        video_T = count_length(test_data)
        C,T,V,M = test_data.shape
        output_result_path = './visualization/{}/ND_{}_{}.mp4'.format(dataset, video_id, video_name.split('.')[0])

        data = torch.from_numpy(test_data)#test_data[:,0:video_T,:,:]
        data = data.unsqueeze(0)
        data = data.float().to(self.dev)

        # extract feature
        print('\nVisualization...', video_name, video_T)
        output, feature = self.part_model.extract_feature(data)
        output = output[0]
        feature = feature[0]  # feature [C,T,V,M] -> [128,T,30,1]
        pred = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)

        # feature heat map
        intensity = (feature * feature).sum(dim=0) ** 0.5  # [T,V,M]
        #intensity = abs(intensity)
        #intensity = intensity / intensity.mean()  # T,V,M
        intensity = intensity / (intensity.max() - intensity.min())

        body_edge = self.part_model.graph.edge
        print('\nNet prediction:', pred)

        if dataset == 'NTU':
            test_data = np.load(pixel_path)[video_id]
            print(test_data.shape)
            pose = test_data
            print('\nLabel:', label, NTU_label_dict['{}'.format(label)])
        else:
            pose = test_data[:,0:video_T,:,:]
            print('\nLabel:', label, SBU_label_dict['{}'.format(label)])

        writer = skvideo.io.FFmpegWriter(output_result_path)# outputdict={'-b': '300000000'}

        for t in range(video_T):
            frame =  np.full((1080, 1920, 3), 255, np.uint8)
            #frame = np.zeros((1080, 1920, 3), np.uint8)

            H, W, c = frame.shape
            scale_factor = 1

            # draw skeleton
            feature_heat = frame
            edge_map = frame
            for m in range(M):
                if dataset == 'NTU':
                    for i, j in body_edge:
                        xi = int(pose[0, t, i+25*m, 0])
                        yi = int(pose[1, t, i+25*m, 0])
                        xj = int(pose[0, t, j+25*m, 0])
                        yj = int(pose[1, t, j+25*m, 0])
                        if i<25 and j<25:
                            cv2.line(edge_map, (xi, yi), (xj, yj), (0, 255, 0), int(np.ceil(4 * scale_factor)))
                        else:
                            cv2.line(edge_map, (xi, yi), (xj, yj), (0, 0, 255), int(np.ceil(4 * scale_factor)))

                    #feature map heat
                    f = (intensity[t // 4, :, 0] ** 5).cpu().detach().numpy()  # intensity [T,V,M]
                    if f.mean() != 0:
                        f = f / f.mean()
                    for v in range(V):
                        x = int(pose[0, t, v+25*m, 0])
                        y = int(pose[1, t, v+25*m, 0])
                        # print('feature map', f)
                        cv2.circle(feature_heat, (x, y), 0, (255, 0, 0),
                                   int(np.ceil(f[v] * 15 * scale_factor)))

                    self.put_text(feature_heat, 'predction: ' + NTU_label_dict['{}'.format(pred)], (0.05, 0.2))  # position ratio (H,W)
                    self.put_text(feature_heat, 'label:' + NTU_label_dict['{}'.format(label)], (0.05, 0.6))

                else:
                    for i, j in body_edge:
                        xi = int(pose[0, t, i, m] * W)
                        yi = int(pose[1, t, i, m] * H)
                        xj = int(pose[0, t, j, m] * W)
                        yj = int(pose[1, t, j, m] * H)
                        cv2.line(edge_map, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil(2 * scale_factor)))

                    for i, j in k_edge:
                        xi = int(pose[0, t, i, m] * W)
                        yi = int(pose[1, t, i, m] * H)
                        xj = int(pose[0, t, j, m] * W)
                        yj = int(pose[1, t, j, m] * H)

                        cv2.line(edge_map, (xi, yi), (xj, yj), (0, 0, 255), int(np.ceil(1 * scale_factor)))

                    f = (intensity[t // 4, :, m]**5).cpu().detach().numpy()  # intensity [T,V,M]
                    if f.mean() != 0:
                        f = f / f.mean()
                    for v in range(V):
                        x = int(pose[0, t, v, m] * W)
                        y = int(pose[1, t, v, m] * H)
                        #print('feature map', f)
                        cv2.circle(feature_heat, (x, y), 0, (255, 0, 0),
                                   int(np.ceil(f[v] * 20 * scale_factor)))

                    self.put_text(feature_heat, 'predction: ' + SBU_label_dict['{}'.format(label)], (0.05, 0.2))  # position ratio (H,W)
                    self.put_text(feature_heat, 'label:' + SBU_label_dict['{}'.format(label)], (0.05, 0.6))

            img = feature_heat + edge_map
            # save video
            writer.writeFrame(img)

        print('visualization Done.')
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))

    def visual_org(self):

        dataset = 'NTU' # SBU / NTU

        #joint
        org_path = self.arg.test_feeder_args['org_data_path']
        pixel_path = './data/NTU_2p_pixel/{}/val_pixel.npy'.format(org_path.split('/')[-2])
        label_path = self.arg.test_feeder_args['label_path']
        video_id = int(self.arg.video_id)

        self.org_model.eval()
        self.org_model.to(self.dev)

        test_data = np.load(org_path)[video_id] # [C, T, V, M] -> [3,T,25,2]
        print(test_data.shape)
        label_data = pk.load(open(label_path, 'rb'))
        video_name = label_data[0][video_id]
        label = label_data[1][video_id]
        
        video_T = count_length(test_data)
        C,T,V,M = test_data.shape
        output_result_path = './visualization/{}/org_{}_{}.mp4'.format(dataset, video_id, video_name.split('.')[0])

        data = torch.from_numpy(test_data)#test_data[:,0:video_T,:,:]
        data = data.unsqueeze(0)
        data = data.float().to(self.dev)

        # extract feature
        print('\nVisualization...', video_name, video_T)
        output, feature = self.org_model.extract_feature(data)
        output = output[0]
        feature = feature[0]  # feature [C,T,V,M] -> [128,T,30,1]
        pred = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)

        # feature heat map
        intensity = (feature * feature).sum(dim=0) ** 0.5  # [T,V,M]
        #intensity = abs(intensity)
        #intensity = intensity / intensity.mean()  # T,V,M
        intensity = intensity / (intensity.max() - intensity.min())

        body_edge = self.org_model.graph.edge
        print('\nNet prediction:', pred)

        if dataset == 'NTU':
            test_data = np.load(pixel_path)[video_id]
            print(test_data.shape)
            pose = test_data
            print('\nLabel:', label, NTU_label_dict['{}'.format(label)])
        else:
            pose = test_data[:,0:video_T,:,:]
            print('\nLabel:', label, SBU_label_dict['{}'.format(label)])

        writer = skvideo.io.FFmpegWriter(output_result_path)# outputdict={'-b': '300000000'}

        for t in range(video_T):
            frame =  np.full((1080, 1920, 3), 255, np.uint8)
            #frame = np.zeros((1080, 1920, 3), np.uint8)

            H, W, c = frame.shape
            scale_factor = 1

            # draw skeleton
            feature_heat = frame
            edge_map = frame
            for m in range(M):
                if dataset == 'NTU':
                    for i, j in body_edge:
                        xi = int(pose[0, t, i+25*m, 0])
                        yi = int(pose[1, t, i+25*m, 0])
                        xj = int(pose[0, t, j+25*m, 0])
                        yj = int(pose[1, t, j+25*m, 0])
                        if m == 0:
                            cv2.line(edge_map, (xi, yi), (xj, yj), (0, 255, 0), int(np.ceil(4 * scale_factor)))
                        else:
                            cv2.line(edge_map, (xi, yi), (xj, yj), (0, 0,255), int(np.ceil(4 * scale_factor)))

                    #feature map heat
                    f = (intensity[t // 4, :, 0] ** 5).cpu().detach().numpy()  # intensity [T,V,M]
                    if f.mean() != 0:
                        f = f / f.mean()
                    for v in range(V):
                        x = int(pose[0, t, v+25*m, 0])
                        y = int(pose[1, t, v+25*m, 0])
                        # print('feature map', f)
                        cv2.circle(feature_heat, (x, y), 0, (255, 0, 0), int(np.ceil(f[v] * 5 * scale_factor)))

                    self.put_text(feature_heat, 'predction: ' + NTU_label_dict['{}'.format(pred)], (0.05, 0.2))  # position ratio (H,W)
                    self.put_text(feature_heat, 'label:' + NTU_label_dict['{}'.format(label)], (0.05, 0.6))

                else:
                    for i, j in body_edge:
                        xi = int(pose[0, t, i, m] * W)
                        yi = int(pose[1, t, i, m] * H)
                        xj = int(pose[0, t, j, m] * W)
                        yj = int(pose[1, t, j, m] * H)
                        cv2.line(edge_map, (xi, yi), (xj, yj), (255, 255, 255), int(np.ceil(2 * scale_factor)))

                    for i, j in k_edge:
                        xi = int(pose[0, t, i, m] * W)
                        yi = int(pose[1, t, i, m] * H)
                        xj = int(pose[0, t, j, m] * W)
                        yj = int(pose[1, t, j, m] * H)

                        cv2.line(edge_map, (xi, yi), (xj, yj), (0, 0, 255), int(np.ceil(1 * scale_factor)))

                    f = (intensity[t // 4, :, m]**5).cpu().detach().numpy()  # intensity [T,V,M]
                    if f.mean() != 0:
                        f = f / f.mean()
                    for v in range(V):
                        x = int(pose[0, t, v, m] * W)
                        y = int(pose[1, t, v, m] * H)
                        #print('feature map', f)
                        cv2.circle(feature_heat, (x, y), 0, (255, 0, 0),
                                   int(np.ceil(f[v] * 20 * scale_factor)))

                    self.put_text(feature_heat, 'predction: ' + SBU_label_dict['{}'.format(label)], (0.05, 0.2))  # position ratio (H,W)
                    self.put_text(feature_heat, 'label:' + SBU_label_dict['{}'.format(label)], (0.05, 0.6))

            img = feature_heat + edge_map
            # save video
            writer.writeFrame(img)

        print('visualization Done.')
        writer.close()
        print('The Demo result has been saved in {}.'.format(output_result_path))

    def wrong_list(self):
        #joint
        joint_result_frag = []
        part_result_frag = []
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
                output_part = self.part_model(part_data)
                output_joint = self.joint_model(joint_data)

            joint_result_frag.append(output_joint.data.cpu().numpy())
            part_result_frag.append(output_part.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

        joint_result = np.concatenate(joint_result_frag)
        part_result = np.concatenate(part_result_frag)
        Label = np.concatenate(label_frag)

        joint_rank = joint_result.argsort()
        joint_wrong = [i for i, l in enumerate(Label) if joint_rank[i, -1] != Label[i]]
        acc_joint =  (joint_result.shape[0]-len(joint_wrong)) / joint_result.shape[0]

        part_rank = part_result.argsort()
        part_wrong = [i for i, l in enumerate(Label) if part_rank[i, -1] != Label[i]]
        acc_part =  (part_result.shape[0]-len(part_wrong)) / part_result.shape[0]

        print(joint_wrong)
        print(part_wrong)
        
        print('Accuracy:', acc_joint, len(joint_wrong), acc_part, len(part_wrong))


    @staticmethod
    def get_parser(add_help=False):
        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='IO Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default='./config/st_gcn/ntu-xsub/visual_demo.yaml', help='path to the configuration file')

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

        parser.add_argument('--org_model', default=None, help='the model will be used')
        parser.add_argument('--org_model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--org_weights', default=None, help='the weights for network initialization')

        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')

        #visualization add
        parser.add_argument('--video_id', type=int, default=43, help='video choose to visualization')
        # endregion yapf: enable

        return parser

if __name__ == '__main__':
    fusion_work = Fusion()
    #fusion_work.start()
    #fusion_work.test_single()
    #fusion_work.wrong_list()

    fusion_work.visual_org()
    #fusion_work.visual_ND()
    #fusion_work.visual()



