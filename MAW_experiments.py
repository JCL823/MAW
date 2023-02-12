'''load packages'''
import tensorflow as tf
import numpy as np
import pickle
import argparse
import os
import time
from copy import deepcopy
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import normalize as nmlz
print(tf.__version__[0])
import MAW_model 
'''Python 3.6.6
   pip install tensorflow==2.0.0b1
   pip install tfp-nightly==0.7.0.dev20190510
   pip install tensorflow_probability==0.8.0rc0 --user --upgrade'''

#%% Parser
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", 
                    help="choose the gpu to use", 
                    default="0")
parser.add_argument("-t", "--true",
                    help="specify normal data (mnist/fashion/kdd/caltech101/reuters5/covid/caltech_mix/caltech_planes/cifar/cifar_affine/cifar_proj/cifar_feat/cifar_mix)",
                    default="mnist")
parser.add_argument("-ntrain", "--ntrain", type=int,
                    help="specify batch size",
                    default=1200) #default = 1200
parser.add_argument("-bs", "--batchsize", type=int,
                    help="specify batch size",
                    default=128) #default = 128
parser.add_argument("-lr", "--learningrate", type=float,
                    help="specify learning rate",
                    default=0.00005) #default = 0.00005
parser.add_argument("-e", "--epochsize", type=int,
                    help="specify number of epoch size",
                    default=1)
parser.add_argument("-r", "--num2run", type=int,
                    help="specify number of runs for each setting",
                    default=1)
parser.add_argument("-samp", "--num_sampling", 
                    help="specify number of sampling",
                    default=1)
parser.add_argument("-nl", "--NormalizationLayer",
                    default="1")
parser.add_argument("-m", "--dim_latent", type=int,
                    help="dimension of latent layer",
                    default=2) #default = 2, (need to be an integer)
parser.add_argument("-l", "--loss",
                    help="specify loss norm type",
                    default="L21")
parser.add_argument("-s", "--score_type",
                    help="specify score type (cosine/L2)",
                    default="cosine")
parser.add_argument("-cval", "--cvalue_train", type=float, 
                    help="specify outlier ratio for training",
                    default=0.1)
parser.add_argument("-cvaltest", "--cvalue_test", type=float, 
                    help="specify outlier ratio for testing",
                    default=0)
parser.add_argument("-mix", "--mix_para", type=float, 
                    help="specify hyperparameter of mixture model",
                    default=0.2) #default = 0.2
parser.add_argument("-puredist", "--puredist", 
                    help="choose whether to generate only from pure dist (1=pure)",
                    default="1")
parser.add_argument("-res", "--result_number", 
                    help="specify result number",
                    default="02")

args = parser.parse_args()

if args.NormalizationLayer == "1":
    NormalizationLayer = True
else:
    NormalizationLayer = False    
     
if args.puredist == "1":
    pure_dist = True
else:
    pure_dist = False    

if args.cvalue_test == 0:
    cvalue_test = (0.1, 0.3, 0.5, 0.7, 0.9)
else:
    cvalue_test = (args.cvalue_test,)


"""Data usage."""
if args.true == "mnist":
    (X_train_origin, y_train_origin), (X_test_origin, y_test_origin) = tf.keras.datasets.mnist.load_data()
elif args.true == "fashion":
    (X_train_origin, y_train_origin), (X_test_origin, y_test_origin) = tf.keras.datasets.fashion_mnist.load_data()
elif args.true == "kdd":
    with open("../data/kdd99.data", "rb") as f:
        data = pickle.load(f)
    X_train_origin, y_train_origin, X_test_origin, y_test_origin = data["X_train"], data["y_train"], data["X_test"], data["y_test"] 
elif args.true == "caltech101":
    with open("../data/caltech101.data", 'rb') as f:
        data = pickle.load(f)
    X_train_origin, y_train_origin, X_test_origin, y_test_origin = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
elif args.true == "reuters5":
    with open("../data/reuters5.data", 'rb') as f:
        data = pickle.load(f)
    X_train_origin, y_train_origin, X_test_origin, y_test_origin = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
elif args.true == "covid":
    with open("../data/covid.data", 'rb') as f:
        data = pickle.load(f)
    X_train_origin, y_train_origin, X_test_origin, y_test_origin = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
elif args.true == "cifar":
    with open("../data/cifar10.data", 'rb') as f:
        data = pickle.load(f)
    X_train_origin, y_train_origin, X_test_origin, y_test_origin = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    y_train_origin = np.reshape(y_train_origin, (y_train_origin.shape[0],))
    y_test_origin = np.reshape(y_test_origin, (y_test_origin.shape[0],))
else:
    raise Exception("Dataset not recognized!")

"""Set GPU for use."""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
     
    
#%%%
if __name__ == "__main__":
    
    num_experiments = args.num2run
    
    if args.true in ("mnist", "fashion", "cifar",):
        anomaly_set = list(range(10))
    elif args.true in ("kdd",):
        anomaly_set = list(range(2))
    elif args.true == "caltech101":
        anomaly_set = list(range(11))
    elif args.true == "reuters5":
        anomaly_set = list(range(5))
    elif args.true in ("covid"):
        anomaly_set = list(range(3))
   
            
    for cvalue in cvalue_test:
        if args.cvalue_train != 0:
            cvalue_train = args.cvalue_train
        else:
            cvalue_train = cvalue

        for anomaly in anomaly_set:
            
            print("Anomaly digit: "+str(anomaly)+"; c: "+str(cvalue))
  
            if args.true in ("mnist", "fashion"):              
                input_shape = (28,28,1)
                num_pure_train = args.ntrain
                num_anomaly_train = int( num_pure_train * cvalue_train)  
                y_train = (np.array(y_train_origin) == anomaly).astype(int)                    
                X_train_normal = X_train_origin[y_train==1][0:num_pure_train]
                X_train_anomaly = X_train_origin[y_train==0][0:num_anomaly_train]
                X_train = np.concatenate((X_train_normal, X_train_anomaly))                    
                X_train = np.reshape(X_train, (-1, 28*28*1)) / 255. * 2 - 1
                X_train = np.reshape(X_train, (-1,28,28,1)).astype("float32") 
                
                y_train_normal = y_train[y_train==1][0:num_pure_train]
                y_train_anomaly = y_train[y_train==0][0:num_anomaly_train]
                y_train = np.concatenate((y_train_normal, y_train_anomaly))                
                y_train = [False] * len(X_train_normal) + [True] * num_anomaly_train
                    
                num_pure = int(num_pure_train/5)
                num_anomaly = int( num_pure * cvalue )
                y_test = (np.array(y_test_origin) == anomaly).astype(int)                    
                X_test_normal = X_test_origin[y_test==1][0:num_pure]
                X_test_anomaly = X_test_origin[y_test==0][0:num_anomaly]
                X_test = np.concatenate((X_test_normal, X_test_anomaly))
                X_test = np.reshape(X_test, (-1, 28*28*1)) / 255. * 2 - 1
                X_test = np.reshape(X_test, (-1,28,28,1)).astype("float32")
                
                y_test_normal = y_test[y_test==1][0:num_pure]
                y_test_anomaly = y_test[y_test==0][0:num_anomaly]
                y_test = np.concatenate((y_test_normal, y_test_anomaly))                
                y_test = [False] * len(X_test_normal) + [True] * num_anomaly                                              

            elif args.true in ("cifar"):
                input_shape= (32, 32, 3)
              
                
                num_pure_train = args.ntrain
                num_anomaly_train = int( num_pure_train * cvalue_train)  
                #X_train_origin = X_train_origin[:,::2,::2,:]
                y_train = (np.array(y_train_origin) == anomaly).astype(int)                    
                X_train_normal = X_train_origin[y_train==1][0:num_pure_train]
                X_train_anomaly = X_train_origin[y_train==0][0:num_anomaly_train]
                X_train = np.concatenate((X_train_normal, X_train_anomaly))                    
                #X_train = np.reshape(X_train, (-1, input_shape[0] * input_shape[1] * input_shape[2])) / 255. * 2 - 1
                X_train = np.reshape(X_train, (-1, input_shape[0] * input_shape[1] * input_shape[2])) / 255.
                X_train = np.reshape(X_train, (-1, input_shape[0], input_shape[1], input_shape[2])).astype("float32") 
                
                y_train_normal = y_train[y_train==1][0:num_pure_train]
                y_train_anomaly = y_train[y_train==0][0:num_anomaly_train]
                y_train = np.concatenate((y_train_normal, y_train_anomaly))                
                y_train = [False] * len(X_train_normal) + [True] * num_anomaly_train
                    
                num_pure = int(num_pure_train/2)
                num_anomaly = int( num_pure * cvalue )
                y_test = (np.array(y_test_origin) == anomaly).astype(int)
                #X_test_origin = X_test_origin[:,::2,::2,:]
                X_test_normal = X_test_origin[y_test==1][0:num_pure]
                X_test_anomaly = X_test_origin[y_test==0][0:num_anomaly]
                X_test = np.concatenate((X_test_normal, X_test_anomaly))
                #X_test = np.reshape(X_test, (-1, input_shape[0] * input_shape[1] * input_shape[2])) / 255. * 2 - 1
                X_test = np.reshape(X_test, (-1, input_shape[0] * input_shape[1] * input_shape[2])) / 255.
                X_test = np.reshape(X_test, (-1, input_shape[0], input_shape[1], input_shape[2])).astype("float32")
                
                y_test_normal = y_test[y_test==1][0:num_pure]
                y_test_anomaly = y_test[y_test==0][0:num_anomaly]
                y_test = np.concatenate((y_test_normal, y_test_anomaly))                
                y_test = [False] * len(X_test_normal) + [True] * num_anomaly                
                
            elif args.true in ("kdd"):
                input_shape = (121,)
                num_pure_train = 6000
                num_anomaly_train = int( num_pure_train * cvalue_train)  
                y_train = (np.array(y_train_origin) == anomaly).astype(int)                    
                X_train_normal = X_train_origin[y_train==1][0:num_pure_train]
                X_train_anomaly = X_train_origin[y_train==0][0:num_anomaly_train]
                X_train = np.concatenate((X_train_normal, X_train_anomaly)).astype("float32")                    
                
                y_train_normal = y_train[y_train==1][0:num_pure_train]
                y_train_anomaly = y_train[y_train==0][0:num_anomaly_train]
                y_train = np.concatenate((y_train_normal, y_train_anomaly))                
                y_train = [False] * len(X_train_normal) + [True] * num_anomaly_train
                    
                num_pure = int(num_pure_train/5)
                num_anomaly = int( num_pure * cvalue )
                y_test = (np.array(y_test_origin) == anomaly).astype(int)                    
                X_test_normal = X_test_origin[y_test==1][0:num_pure]
                X_test_anomaly = X_test_origin[y_test==0][0:num_anomaly]
                X_test = np.concatenate((X_test_normal, X_test_anomaly)).astype("float32")
                
                y_test_normal = y_test[y_test==1][0:num_pure]
                y_test_anomaly = y_test[y_test==0][0:num_anomaly]
                y_test = np.concatenate((y_test_normal, y_test_anomaly))                
                y_test = [False] * len(X_test_normal) + [True] * num_anomaly                                        
          
                
            elif args.true in ("caltech101"):
                input_shape = (32,32,3)
                num_pure_train = 100
                num_pure = 100
                                           
                num_anomaly_train = int(num_pure_train * cvalue_train)  
                y_train = (np.array(y_train_origin) == anomaly).astype(int)                    
                X_train_normal = X_train_origin[y_train==1][0:num_pure_train]
                X_train_anomaly = X_train_origin[y_train==0][0:num_anomaly_train]
                X_train = np.concatenate((X_train_normal, X_train_anomaly))                    
                X_train = np.reshape(X_train, (-1, 32*32*3)) / 255. * 2 - 1
                X_train = nmlz(X_train)
                X_train = np.reshape(X_train, (-1,32,32,3)).astype("float32")
                
                y_train_normal = y_train[y_train==1][0:num_pure_train]
                y_train_anomaly = y_train[y_train==0][0:num_anomaly_train]
                y_train = np.concatenate((y_train_normal, y_train_anomaly))                
                y_train = [False] * len(X_train_normal) + [True] * num_anomaly_train
                                    
                num_anomaly = int( num_pure * cvalue )
                y_test = (np.array(y_test_origin) == anomaly).astype(int)                    
                X_test_normal = X_test_origin[y_test==1][0:num_pure]
                X_test_anomaly = X_test_origin[y_test==0][0:num_anomaly]
                X_test = np.concatenate((X_test_normal, X_test_anomaly))
                X_test = np.reshape(X_test, (-1, 32*32*3)) / 255. * 2 - 1
                X_test = nmlz(X_test)
                X_test = np.reshape(X_test, (-1,32,32,3)).astype("float32")
                
                y_test_normal = y_test[y_test==1][0:num_pure]
                y_test_anomaly = y_test[y_test==0][0:num_anomaly]
                y_test = np.concatenate((y_test_normal, y_test_anomaly))                
                y_test = [False] * len(X_test_normal) + [True] * num_anomaly

            elif args.true in ("reuters5"):
                input_shape = (26147,)
                num_pure_train = 350
                num_anomaly_train = int( num_pure_train * cvalue_train)  
                y_train = (np.array(y_train_origin) == anomaly).astype(int)                    
                X_train_normal = X_train_origin[y_train==1][0:num_pure_train]
                X_train_anomaly = X_train_origin[y_train==0][0:num_anomaly_train]
                X_train = np.concatenate((X_train_normal, X_train_anomaly)).astype("float32")                    
                
                y_train_normal = y_train[y_train==1][0:num_pure_train]
                y_train_anomaly = y_train[y_train==0][0:num_anomaly_train]
                y_train = np.concatenate((y_train_normal, y_train_anomaly))                
                y_train = [False] * len(X_train_normal) + [True] * num_anomaly_train
                    
                num_pure = 140
                num_anomaly = int( num_pure * cvalue )
                y_test = (np.array(y_test_origin) == anomaly).astype(int)                    
                X_test_normal = X_test_origin[y_test==1][0:num_pure]
                X_test_anomaly = X_test_origin[y_test==0][0:num_anomaly]
                X_test = np.concatenate((X_test_normal, X_test_anomaly)).astype("float32")
                
                y_test_normal = y_test[y_test==1][0:num_pure]
                y_test_anomaly = y_test[y_test==0][0:num_anomaly]
                y_test = np.concatenate((y_test_normal, y_test_anomaly))                
                y_test = [False] * len(X_test_normal) + [True] * num_anomaly

            elif args.true in ("covid"):
                input_shape= (64, 64, 3)
                num_pure_train = 160
                num_anomaly_train = int( num_pure_train * cvalue_train)  
                y_train = (np.array(y_train_origin) == anomaly).astype(int)   

                num_orig_normal = X_train_origin[y_train==1].shape[0]
                num_orig_anomaly = X_train_origin[y_train==0].shape[0]
                select_normal = np.random.randint(num_orig_normal, size=num_pure_train)
                select_anomaly = np.random.randint(num_orig_anomaly, size=num_anomaly_train)
                
                X_train_normal = X_train_origin[y_train==1][0:num_pure_train]
                X_train_anomaly = X_train_origin[y_train==0][0:num_anomaly_train]                    
                y_train_normal = y_train[y_train==1][0:num_pure_train]
                y_train_anomaly = y_train[y_train==0][0:num_anomaly_train]
                
                X_train = np.concatenate((X_train_normal, X_train_anomaly))                    
                X_train = np.reshape(X_train, (-1, input_shape[0] * input_shape[1] * input_shape[2])) / 255. * 2 - 1
                X_train = np.reshape(X_train, (-1, input_shape[0], input_shape[1], input_shape[2])).astype("float32") 

                y_train = np.concatenate((y_train_normal, y_train_anomaly))                
                y_train = [False] * len(X_train_normal) + [True] * num_anomaly_train
                    
                num_pure = 59
                num_anomaly = int( num_pure * cvalue )
                y_test = (np.array(y_test_origin) == anomaly).astype(int)                    
                X_test_normal = X_test_origin[y_test==1][0:num_pure]
                X_test_anomaly = X_test_origin[y_test==0][0:num_anomaly]
                X_test = np.concatenate((X_test_normal, X_test_anomaly))
                X_test = np.reshape(X_test, (-1, input_shape[0] * input_shape[1] * input_shape[2])) / 255. * 2 - 1
                X_test = np.reshape(X_test, (-1, input_shape[0], input_shape[1], input_shape[2])).astype("float32")
                
                y_test_normal = y_test[y_test==1][0:num_pure]
                y_test_anomaly = y_test[y_test==0][0:num_anomaly]
                y_test = np.concatenate((y_test_normal, y_test_anomaly))                
                y_test = [False] * len(X_test_normal) + [True] * num_anomaly   

            X = deepcopy(X_train[:])
            np.random.shuffle(X)
            Y = y_train           

            if args.true in ("mnist", "fashion", "caltech101", "covid", "cifar"):
                activation = tf.nn.tanh
            else:
                activation = tf.nn.sigmoid

            aucs = []
            aps = []
            test_aucs=[]
            test_aps=[]          
            time_elapses = []
            
            for idx_exp in range(num_experiments):
                print(f"Experiment No. {idx_exp+1}")
                '''Define the network architecture'''  

                           
                model = MAW_model.GMM(  
                        lr_rsr = args.learningrate, 
                        latent_loss_div=1, 
                        epoch_size = args.epochsize, 
                        batch_size = args.batchsize,
                        ipt_shape = input_shape,
                        activation = activation,
                        hidden_layer_sizes = (32,64,128),
                        intrinsic_size = args.dim_latent,
                        norm_type='L21', 
                        loss_norm_type = args.loss,
                        NormalizationLayer = NormalizationLayer,
                        num_sampling = args.num_sampling,
                        gradient_penalty_weight = 10.0,
                        lambda1 = 0.01,
                        lambda2 = 0.01,
                        prior_c = args.mix_para,
                        pure_dists = pure_dist,
                        )
                tStart = time.time()
                model.fit(X)
               
                tEnd = time.time()
                tDiff = tEnd - tStart
                
                scores = []
                for _ in range(int(args.num_sampling)): 
                    features = model.get_reconstruction(X)
                    flat_output = np.reshape(features, (np.shape(X)[0], -1))
                    flat_input = np.reshape(X, (np.shape(X)[0], -1))
                    if args.score_type == "cosine":
                        score = np.sum(flat_output * flat_input, -1) / (np.linalg.norm(flat_output, axis=-1) + 0.000001) / (np.linalg.norm(flat_input, axis=-1) + 0.000001)
                    elif args.score_type == "L2":
                        score = np.linalg.norm( flat_output - flat_input, ord=2, axis=1) 
                    elif args.score_type == "cauchy":
                        score = np.log( 1 + np.square ( np.linalg.norm( flat_output - flat_input, ord=2, axis=1) ))
                    scores.append(score)   
                    
                if args.score_type == "cosine": 
                    similarity = -np.mean(scores, axis=0)
                elif args.score_type in ("L2", "cauchy"): 
                    similarity = np.mean(scores, axis=0)
                else:
                    raise Exception("Similarity measurement type error!")  
            
                    
                auc = roc_auc_score(Y, similarity)
                ap = average_precision_score(Y, similarity)
                                    
                print("auc = ", auc)
                print("ap = ", ap)
                print("time elapse = ", tDiff)
                aucs.append(auc)
                aps.append(ap)
                time_elapses.append(tDiff)
    
                std_auc = np.std(aucs)
                std_ap = np.std(aps)
                std_time = np.std(time_elapses)
    
                test_scores = [] 
                test_features = []                  
                for _ in range(int(args.num_sampling)): 
                    test_feature = model.get_reconstruction(X_test)
                    test_flat_output = np.reshape(test_feature, (np.shape(X_test)[0], -1))
                    test_flat_input = np.reshape(X_test, (np.shape(X_test)[0], -1))
                    if args.score_type == "cosine":                    
                        test_score = np.sum(test_flat_output * test_flat_input, -1) / (np.linalg.norm(test_flat_output, axis=-1) + 0.000001) / (np.linalg.norm(test_flat_input, axis=-1) + 0.000001)
                    elif args.score_type == "L2":   
                        test_score = np.linalg.norm( test_flat_output - test_flat_input, ord=2, axis=1)
                    elif args.score_type == "cauchy":
                        test_score = np.log( 1 + np.square ( np.linalg.norm( test_flat_output - test_flat_input, ord=2, axis=1) )) 


                    test_scores.append(test_score)
                    test_features.append(test_feature)    
                
                if args.score_type == "cosine":
                    test_similarity = -np.mean(test_scores, axis=0)
                elif args.score_type in ("L2", "cauchy"):
                    test_similarity = np.mean(test_scores, axis=0)  

                

