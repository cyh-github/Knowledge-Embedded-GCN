import numpy as np
import os



def single2two(file):
	f_single = np.load(file)
	N,C,T,V,M = f_single.shape
	f_two = np.zeros((N,C,T,V*M,1))
	f_two[:,:,:,:,0] = np.concatenate((f_single[:,:,:,:,0], f_single[:,:,:,:,1]),axis=-1)
	return f_two

if __name__ == '__main__':
	save_path = '/home/f1y/cyh/st-gcn-li/data/SBU_2p'
	for mode in ['train', 'val']:
		for set_n in ['set1', 'set2', 'set3', 'set4', 'set5']:
			data_npy = '/home/f1y/cyh/st-gcn-li/data/SBU_single/{}_data_{}.npy'.format(mode, set_n)
			data_name = data_npy.split('/')[-1]
			save_file = os.path.join(save_path, data_name)
			f_two = single2two(data_npy)
			np.save(save_file, f_two)
	
