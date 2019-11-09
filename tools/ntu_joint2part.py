import numpy as np

joint = 25
part_num = 8*2
left_arm = [8,9,10] #part 1
left_hand = [11,23,24]#part 2
left_leg = [16,17,18]#part 3
right_arm = [4,5,6]#part 4
right_hand = [7,21,22]#part 5
right_leg = [12,13,14]#part 6
head = [2,3,20]#part 7
trunk = [0,1,20]#part 8

#joint data load
data_org = np.load('/home/f1y/cyh/st-gcn-3A/data/NTU_2p_data/xsub/train_data_2per.npy')
N,C,T,V,M = data_org.shape

#person1 parts prepare
P1_left_arm = np.concatenate((data_org[:,:,:,left_arm[0],:],data_org[:,:,:,left_arm[1],:],data_org[:,:,:,left_arm[2],:]),axis=1)[:,:,:,np.newaxis,:]
P1_left_hand = np.concatenate((data_org[:,:,:,left_hand[0],:],data_org[:,:,:,left_hand[1],:],data_org[:,:,:,left_hand[2],:]),axis=1)[:,:,:,np.newaxis,:]
P1_left_leg = np.concatenate((data_org[:,:,:,left_leg[0],:],data_org[:,:,:,left_leg[1],:],data_org[:,:,:,left_leg[2],:]),axis=1)[:,:,:,np.newaxis,:]
P1_right_arm = np.concatenate((data_org[:,:,:,right_arm[0],:],data_org[:,:,:,right_arm[1],:],data_org[:,:,:,right_arm[2],:]),axis=1)[:,:,:,np.newaxis,:]
P1_right_hand = np.concatenate((data_org[:,:,:,right_hand[0],:],data_org[:,:,:,right_hand[1],:],data_org[:,:,:,right_hand[2],:]),axis=1)[:,:,:,np.newaxis,:]
P1_right_leg = np.concatenate((data_org[:,:,:,right_leg[0],:],data_org[:,:,:,right_leg[1],:],data_org[:,:,:,right_leg[2],:]),axis=1)[:,:,:,np.newaxis,:]
P1_head = np.concatenate((data_org[:,:,:,head[0],:],data_org[:,:,:,head[1],:],data_org[:,:,:,head[2],:]),axis=1)[:,:,:,np.newaxis,:]
P1_trunk = np.concatenate((data_org[:,:,:,trunk[0],:],data_org[:,:,:,trunk[1],:],data_org[:,:,:,trunk[2],:]),axis=1)[:,:,:,np.newaxis,:]

P1 = np.concatenate((P1_left_arm,P1_left_hand,P1_left_leg,P1_right_arm,P1_right_hand,P1_right_leg,P1_head,P1_trunk),axis=3)#person1 all part

#person2 parts prepare
P2_left_arm = np.concatenate((data_org[:,:,:,left_arm[0]+joint,:],data_org[:,:,:,left_arm[1]+joint,:],data_org[:,:,:,left_arm[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]
P2_left_hand = np.concatenate((data_org[:,:,:,left_hand[0]+joint,:],data_org[:,:,:,left_hand[1]+joint,:],data_org[:,:,:,left_hand[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]
P2_left_leg = np.concatenate((data_org[:,:,:,left_leg[0],:]+joint,data_org[:,:,:,left_leg[1]+joint,:],data_org[:,:,:,left_leg[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]
P2_right_arm = np.concatenate((data_org[:,:,:,right_arm[0]+joint,:],data_org[:,:,:,right_arm[1]+joint,:],data_org[:,:,:,right_arm[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]
P2_right_hand = np.concatenate((data_org[:,:,:,right_hand[0]+joint,:],data_org[:,:,:,right_hand[1]+joint,:],data_org[:,:,:,right_hand[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]
P2_right_leg = np.concatenate((data_org[:,:,:,right_leg[0]+joint,:],data_org[:,:,:,right_leg[1]+joint,:],data_org[:,:,:,right_leg[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]
P2_head = np.concatenate((data_org[:,:,:,head[0]+joint,:],data_org[:,:,:,head[1]+joint,:],data_org[:,:,:,head[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]
P2_trunk = np.concatenate((data_org[:,:,:,trunk[0]+joint,:],data_org[:,:,:,trunk[1]+joint,:],data_org[:,:,:,trunk[2]+joint,:]),axis=1)[:,:,:,np.newaxis,:]

P2 = np.concatenate((P2_left_arm,P2_left_hand,P2_left_leg,P2_right_arm,P2_right_hand,P2_right_leg,P2_head,P2_trunk),axis=3)#person2 all part

data_new = np.concatenate((P1,P2),axis=3)
#path save the part data
np.save('/home/f1y/cyh/st-gcn-3A/data/NTU_2p_part/xsub/train_data_2per.npy',data_new)
print(data_new.shape)
print('finish')

