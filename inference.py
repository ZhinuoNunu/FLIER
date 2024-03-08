import pickle

with open('/home/prp2/zzn/data/imagenet/images/imagenet_v2_test.pkl','rb') as f:
    aa = pickle.load(f)
    
with open('/home/prp2/zzn/data/imagenet/images/imagenet_v2_test_3.pkl','wb') as f:
    pickle.dump(aa,f,protocol=3)