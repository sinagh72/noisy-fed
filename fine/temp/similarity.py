from numpy.linalg import eigh
from numpy.linalg import eig
from scipy.stats import gmean

import numpy as np

def train_x_y(train_dataset, dataset_name):
    #mat = [np.array(x[0]) for x in train_dataset];
    if dataset_name== "SVHN":
        # mat = [np.array(x[0]).reshape((3*32*32)) for x in train_dataset]
        hog_features = np.load('./hog_features.npy')
        print ('SVHN Training data shape: ', hog_features.shape)
        return hog_features, []

    elif dataset_name == "CIFAR10":
        hog_features = np.load('./hog_features_cifar10.npy')
        print ('CIFAR10 Training data shape: ', hog_features.shape)
        return hog_features, []

    elif dataset_name == "CIFAR100":
        hog_features = np.load('./hog_features_cifar100.npy')
        print ('CIFAR100 Training data shape: ', hog_features.shape)
        return hog_features, []


    elif dataset_name== "FashionMNIST" or dataset_name== "MNIST":
        mat = [np.array(x[0]).reshape((28*28)) for x in train_dataset]

        X_train=np.stack(mat)
        y_train=np.stack([np.array(x[1]) for x in train_dataset])
        print (dataset_name+' Training data shape: ', X_train.shape)
        print (dataset_name+' Training labels shape: ', y_train.shape)

        return X_train,y_train




def train_users(tr_idcs,X_train):
    tr_idcs = [int(x) for x in list(tr_idcs)]
    return X_train[tr_idcs,:]

def eig_eval(data):
    #user=user.reshape(user.shape[0],3072) #extra because CIFAr is 3D
    values,vectors=eigh((1/data.shape[0])*data.T@data)
    #values,vectrs=eig((1/user.shape[0])*user.T@user)
    return values, vectors
    
def eig_estimate(data, eig_vector):
    #user=user.reshape(user.shape[0],3072) #extra because CIFAr is 3D
    estimated_eigs=np.linalg.norm(((1/data.shape[0])*data.T@data)@eig_vector,axis=0)
    return estimated_eigs

def div_rel(true_eigs, estimated_eigs, threshold):
#     print(f"true_eigs is {true_eigs}")
#     print(f"estimated_eigs is {estimated_eigs}")
    th=np.minimum(len(np.where(true_eigs>threshold)[0]),len(np.where(estimated_eigs>threshold)[0]))
    clpd_true_eigs,clpd_estimated_eigs=true_eigs[::-1][:th],estimated_eigs[::-1][:th]
    Div=gmean(np.abs(clpd_true_eigs-clpd_estimated_eigs)/np.maximum(clpd_true_eigs,clpd_estimated_eigs))
    Rel=gmean(np.minimum(clpd_true_eigs,clpd_estimated_eigs)/np.maximum(clpd_true_eigs,clpd_estimated_eigs))
    return Div, Rel

##################
def get_eigens_users(X_train,num_workers,worker_train_idx):
    user_eigens={}
    for usrx_id in range(num_workers):
        user_x=train_users(worker_train_idx[usrx_id],X_train)
        user_eigens[usrx_id]=eig_eval(user_x)
    return user_eigens
#######################
def div_rel_matrix_u2u_v3(num_workers,worker_train_idx,cut_thr, X_train):


    div_all=np.zeros((num_workers,num_workers))
    rel_all=np.zeros((num_workers,num_workers))
    user_eigens = get_eigens_users(X_train, num_workers, worker_train_idx ) #eigen_value,eigen vector
    for usrx_id in range(num_workers):
        for usry_id in range(num_workers):
            if usrx_id==usry_id:
                rel_all[usrx_id,usry_id]=1
                rel_all[usry_id,usrx_id]=1
                div_all[usrx_id,usry_id]=0
                div_all[usry_id,usrx_id]=0
            else:

                user_x=train_users(worker_train_idx[usrx_id],X_train)
                user_y=train_users(worker_train_idx[usry_id],X_train)

#                 eigvalues_x,eigvectors_x=eig_eval(user_x) # also fail with ray
#                 eigvalues_y,eigvectors_y=eig_eval(user_y)

#                 estimated_eigs=eig_estimate(user_x,eigvectors_y) # new threshold
                estimated_eigs=eig_estimate(user_x,user_eigens[usry_id][1]) # new threshold

                div_y,rel_y=div_rel(user_eigens[usrx_id][0],estimated_eigs,cut_thr)

                estimated_eigs=eig_estimate(user_y,user_eigens[usrx_id][1]) #new threshold

                div_x,rel_x=div_rel(user_eigens[usry_id][0],estimated_eigs,cut_thr)
                
                
                div_all[usrx_id,usry_id]=(div_x+div_y)/2
                rel_all[usrx_id,usry_id]=(rel_x+rel_y)/2
                div_all[usry_id,usrx_id]=div_all[usrx_id,usry_id]
                rel_all[usry_id,usrx_id]=rel_all[usrx_id,usry_id]
    return div_all,rel_all
###############################