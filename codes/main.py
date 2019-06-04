#nv python3





from __future__ import division, print_function
from collections import defaultdict
import os, pickle, sys
import shutil
from functools import partial

import cv2
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy.misc import imresize
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist

# from models import *
# from metrics import dice_coef, dice_coef_loss
from augmenters import *

import matplotlib.pyplot as plt
import utils
from keras.callbacks import TensorBoard
import newmodels
import losses


def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def data_to_array(img_rows, img_cols):

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows/8),int(img_cols/8)) )

    fileList =  os.listdir('../data/train/')

    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList = list(fileList)
    fileList.sort()

    val_list = [5,15,25,35,45]
    train_list = list( set(range(50)) - set(val_list) )
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []

        filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in the_list), fileList)

        for filename in filtered:

            itkimage = sitk.ReadImage('../data/train/'+filename)
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append( imgs )

            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs )

        images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
        masks = masks.astype(int)

        #Smooth images using CurvatureFlow
        images = smooth_images(images)

        if count==0:
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu)/sigma

            np.save('../data/X_train.npy', images)
            np.save('../data/y_train.npy', masks)
        elif count==1:
            images = (images - mu)/sigma

            np.save('../data/X_val.npy', images)
            np.save('../data/y_val.npy', masks)
        count+=1

    fileList =  os.listdir('../data/test/')
    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList = list(fileList)
    fileList.sort()
    n_imgs=[]
    images=[]
    for filename in fileList:
        itkimage = sitk.ReadImage('../data/test/'+filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        images.append(imgs)
        n_imgs.append( len(imgs) )

    images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
    images = smooth_images(images)
    images = (images - mu)/sigma
    np.save('../data/X_test.npy', images)
    np.save('../data/test_n_imgs.npy', np.array(n_imgs) )


def load_data():

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')


    return X_train, y_train, X_val, y_val #shape (1250, 256, 256, 1) (1250, 256, 256, 1) (127, 256, 256, 1) (127, 256, 256, 1)



def augment_validation_data(X_train, y_train, seed=10):

    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]

    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows*1.5, sigma=img_rows*0.07 )
    # we create two instances with the same arguments
    data_gen_args = dict(preprocessing_function=elastic)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(X_train, seed=seed)
    mask_datagen.fit(y_train, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=100, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=100, seed=seed)

    train_generator = zip(image_generator, mask_generator)

    count=0
    X_val = []
    y_val = []

    for X_batch, y_batch in train_generator:

        if count==5:
            break

        count+=1

        X_val.append(X_batch)
        y_val.append(y_batch)

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    return X_val, y_val


def keras_fit_generator_attention(img_rows=96, img_cols=96, n_imgs=10**4, batch_size=32, regenerate=True, model_type = 'unet', loss_type='dice', train=True, test=True):

    if regenerate:
        data_to_array(img_rows, img_cols)
        #preprocess_data()

    X_train, y_train, X_val, y_val = load_data()
    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]

    
    gt1 = y_train[:,::8,::8,:]
    gt2 = y_train[:,::4,::4,:]
    gt3 = y_train[:,::2,::2,:]
    gt4 = y_train
    gt_train = [gt1,gt2,gt3,gt4]
    
    #choose loss function
    if loss_type == 'dice':
        loss_f = losses.dice_loss
    elif loss_type =='tversky':
        loss_f = losses.tversky_loss
    elif loss_type =='focal_tversky':
        loss_f = losses.focal_tversky
    else:
        print('wrong loss function type')
        return -1
    
    plot_type = 0
    epochs_num = 50
    model_name = model_type+'_'+loss_type
    filepath='../data/weights/weights_'+model_type+'_'+loss_type+'.hdf5'  #you need to create this folder before run
    result_text_path='../data/results/results.txt'  #you need to create this file before run
    result_images_path='../data/results/images1/'   #you need to create this folder before run
    #choose model
    if model_type=='unet':
        sgd = SGD(lr=0.01, momentum=0.90)
        model = newmodels.unet(sgd, (256,256,1), loss_f)
        model_checkpoint = ModelCheckpoint(filepath, monitor='val_dsc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='max')
    elif model_type=='attn_unet':
        sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
        model = newmodels.attn_unet(sgd, (256,256,1), loss_f)
        model_checkpoint = ModelCheckpoint(filepath, monitor='val_dsc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='max')
    elif model_type=='ds_mi_attn_unet':
        plot_type = 1
        y_train = gt_train
        sgd = SGD(lr=0.01, momentum=0.90, decay=1e-6)
        model = newmodels.attn_reg(sgd,(256,256,1),loss_f)
        model_checkpoint = ModelCheckpoint(filepath, monitor='val_final_dsc', 
                             verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='max')
    else:
        print('wrong model type')
        return -1

    model.summary()

    c_backs = [model_checkpoint]
    c_backs.append( EarlyStopping(monitor='loss', min_delta=0.001, patience=5) )
    
    
    model_name = (model_type+'_'+loss_type+'{}').format(int(time.time()))
    tb_call_back = TensorBoard(log_dir='./log_dir_5.25.1/{}'.format(model_name))
    c_backs.append(tb_call_back)

    if train:
        hist = model.fit(X_train, y_train, validation_split=0.15,
                     shuffle=True, epochs=epochs_num, batch_size=batch_size,
                     verbose=True, callbacks=c_backs)#, callbacks=[estop,tb])

        h = hist.history
#         utils.plot(h, epochs_num, batch_size, img_cols, plot_type, model_name = model_name)

    if test==True:
        X_val = np.load('../data/X_val.npy')
        y_val = np.load('../data/y_val.npy')
        num_test = X_val.shape[0]
        test_img_list = os.listdir('../data/test/')
        if model_type=='ds_mi_attn_unet':
            _,_,_,preds = model.predict(X_val)
        else:
            preds = model.predict(X_val)   #use this if the model is not muti-input ds unet
        
        preds_up=[]
        dsc = np.zeros((num_test,1))
        recall = np.zeros_like(dsc)
        tn = np.zeros_like(dsc)
        prec = np.zeros_like(dsc)

        thresh = 0.5
        
        
        # check the predictions from the trained model 
        for i in range(num_test):
            gt = y_val[i]
            pred_up = cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            preds_up.append(pred_up)
            dsc[i] = utils.check_preds(pred_up > thresh, gt)
            recall[i], _, prec[i] = utils.auc(gt, pred_up >thresh)

        f = open(result_text_path, "a")
        f.write('\n')
        f.write('-'*30)
        f.write('\nModel name: ')
        f.write(model_name)
        f.write('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
                np.sum(dsc)/num_test,  
                np.sum(recall)/num_test,
                np.sum(prec)/num_test ))
        f.write('\n')
        f.close()
        
        
        print('-'*30)
        print('At threshold =', thresh)
        print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
                np.sum(dsc)/num_test,  
                np.sum(recall)/num_test,
                np.sum(prec)/num_test ))

        # check the predictions with the best saved model from checkpoint
        model.load_weights(filepath)
    
        if model_type=='ds_mi_attn_unet':
            _,_,_,preds = model.predict(X_val)
        else:
            preds = model.predict(X_val)   #use this if the model is not muti-input ds unet

        preds_up=[]
        dsc = np.zeros((num_test,1))
        recall = np.zeros_like(dsc)
        tn = np.zeros_like(dsc)
        prec = np.zeros_like(dsc)

        for i in range(num_test):
            gt = y_val[i]
            pred_up = cv2.resize(preds[i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            preds_up.append(pred_up)
            dsc[i] = utils.check_preds(pred_up > thresh, gt)
            recall[i], _, prec[i] = utils.auc(gt, pred_up >thresh)

        print('-'*30)
        print('USING HDF5 saved model at thresh=', thresh)
        print('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
                np.sum(dsc)/num_test,  
                np.sum(recall)/num_test,
                np.sum(prec)/num_test ))
        
        f = open(result_text_path, "a")
        f.write('\n')
        f.write('-'*30)
        f.write('\nModel name: ')
        f.write(model_name)
        f.write('\nUSING HDF5 saved model')
        f.write('\n DSC \t\t{0:^.3f} \n Recall \t{1:^.3f} \n Precision\t{2:^.3f}'.format(
                np.sum(dsc)/num_test,  
                np.sum(recall)/num_test,
                np.sum(prec)/num_test ))
        f.write('\n')
        f.close()

        while True:
            idx = np.random.randint(0,num_test)
            if utils.avg_img(y_val[idx]>0)>3 and utils.avg_img(y_val[idx]>0) <8:
#                 print(utils.avg_img(y_val[idx]>0))
                break

        idxs = [94,55,69,75,52,77]
        
        for idx in idxs:
            #plot a test sample for each model
            gt_plot = y_val[idx]
            plt.figure(dpi=200)
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(np.squeeze(gt_plot), cmap='gray')
            plt.title('Original Segmentated Img {}'.format(idx))
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(np.squeeze(preds_up[idx]), cmap='gray')
            plt.title('Mask {}'.format(idx))

            plt.savefig(result_images_path+str(idx)+'/'+model_name+'ori-gt-'.format('.png'))
   
    

if __name__=='__main__':

    CUDA_CACHE_PATH='~/yining/CUDA_CACHE'
    import time

    start = time.time()

    model_types = ['unet','attn_unet','ds_mi_attn_unet']
    loss_types = ['dice','tversky','focal_tversky']
    
    for i in range(3):
#         j=1
        for j in range(3):
            keras_fit_generator_attention(img_rows=256, img_cols=256, regenerate=False,model_type = model_types[i], loss_type=loss_types[j], train = False,test = True,n_imgs=15*10**4, batch_size=16)

    
#     X_train, y_train, X_val, y_val = load_data()
#     y_total = np.append(y_train,y_val,axis=0)
#     por = 0
    
#     for idx in range(y_total.shape[0]):
#         por += utils.avg_img(y_total[idx]>0)
    
#     por /= y_total.shape[0]
#     print(por)
    
    end = time.time()

    print('Elapsed time:', round((end-start)/60, 2 ) )
