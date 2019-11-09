# Knowledge-Embedded-GCN  
A pytorch implementation of Knowledge-Embedded-GCN  
***  
**Environment**  
-pytorch 0.4.1  
-python 3.7.0  
More detials version information please refer in `requirements.txt` 
  
**Install**  
Run `pip install -r requirements.txt` to quickly install all pickages need.  

**Our key block**  
![network architecture](https://github.com/cyh-github/Knowledge-Embedded-GCN/blob/master/fig/3A_block.png)
  
**data process**  
Here we take NTU dataset as an example:
Step1. Download NTU dataset(contain RGB and skeleton) from: [NTU dataset][http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp]， here we only need the skeleton data. Save the download "nturgb+d_skeletons" at `./data/NTU-RGB-D/nturgb+d_skeletons`.  
Step2. Read the skeleton data from `nturgb+d_skeletons` as the format we need.  Here we `cd master directory` and run `python ./tools/ntu_generate.py` to  generate the train/test data and label files. Note that these are joint data.    
Step3. Create the part data from joint data we obtained at step2. Run `python ./tools/ntu_joint2part.py` to get part data.  
From above process, we get all data format to go on our experiments.


**train**  
Use `python main.py recognition -c ConfigFile` start training. ConfigFile is the train config yaml file we provide in `./config`.  
Note that some items need to be changed depend on your specific path: For example `train_feeder_args` and `test_feeder_args`.  

**test**  
If you want to test directly without training, here we provide our [pretrained model][https://pan.baidu.com/s/1TGfIk61TX18OZS_N_uFnpA]  
  
*single model test*:  
Run `python main.py recognition -c ConfigFile` to test single model either NTU or SBU model. ConfigFile is the test config yaml file we provide in `./config`. 
  
*latefusion of joint and part model*:  
Run `python latefusion.py -c ConfigFile` to fuse the joint model and part model. ConfigFile is the fusion config yaml file we provide in `./config`.  



