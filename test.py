import torch

from cgi import test
from distutils.command.build_ext import build_ext
import os, time
from pyexpat import model
from operator import add
from venv import create
import numpy as np
from glob import glob
import cv2
from sklearn import metrics
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score
from zmq import device
from patchify import patchify, unpatchify

from model import build_unet
from utils import create_dir, seeding
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from utils import epoch_time, get_image_names, get_paths, seeding


def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    #y_true = y_true.cpu().numpy()
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    #y_pred = y_pred.cpu().numpy()
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    #score_jaccard = jaccard_score(y_true, y_pred, zero_division = 1, average="micro")
    #score_f1 = f1_score(y_true, y_pred, zero_division = 1)
    #score_recall = recall_score(y_true, y_pred, zero_division = 1)
    #score_precision = precision_score(y_true, y_pred, zero_division = 1)
    #score_acc = accuracy_score(y_true, y_pred)

    #return [score_jaccard, score_f1, score_recall, score_precision, score_acc]

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n")
    sns.heatmap(cm, annot=True)
    plt.show()

    print(cm)

    backround_IoU = cm[0][0] / (cm[0][0] + cm[0][1] + cm[0][2])
    GBM_IoU = cm[1][1] / (cm[1][0] + cm[1][1] + cm[1][2])
    Podocyte_IoU = cm[2][2] / (cm[2][0] + cm[2][1] + cm[2][2])

    print("Background IoU:", backround_IoU)
    print("GBM IoU:", GBM_IoU)
    print("Podocyte IoU:", Podocyte_IoU)
    print("Mean IoU:", (backround_IoU + GBM_IoU + Podocyte_IoU) / 3)
    

    return [cm, backround_IoU, GBM_IoU, Podocyte_IoU]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask







LABEL_TO_COLOR = {0:[0,0,0], 1:[0,255,0], 2:[255,0,0]}

"""RGB to grayscale mask"""
def rgb2mask(rgb):
    
    mask = np.zeros((rgb.shape[0], rgb.shape[1]))

    for k, v in LABEL_TO_COLOR.items():
        mask[np.all(rgb==v, axis=2)] = k
        
    return mask

def mask2rgb(mask):
    
    rgb = np.zeros(mask.shape+(3,), dtype=np.uint8)
    
    for i in np.unique(mask):
            rgb[mask==i] = LABEL_TO_COLOR[i]
            
    return rgb



if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")
    create_dir("results_no_patches")

    """ Load dataset """
    # Loading the test images
    # in this case validation is the same with test images
    # test_x = sorted(glob("../new_data/test/image/*"))
    # test_y = sorted(glob("../new_data/test/mask/*"))

    path = "C:/Users/Aristotelis/src/Thesis_src/dataset"

    images = np.array(sorted(glob(os.path.join(path, "images", "*.tif"))))
    masks = np.array(sorted(glob(os.path.join(path, "masks_rgb", "*.png"))))

    print(f"Number of Images: {len(images)}")
    print(f"Number of RGB Masks: {len(masks)}")




    """ K - Fold Split (5 - Fold) """

    # prepare cross validation
    kfold = KFold(n_splits=5)

    i = 0
    # enumerate splits
    for tmp_train, tmp_test in kfold.split(images):
        train_set, valid_set = train_test_split(tmp_train, test_size=0.3, random_state=17)
        test_set = tmp_test
        print("----------------------------------------------------------")
        print("Fold", i + 1)
        print("Train:", train_set, "Valid:", valid_set, "Test: ", test_set)
        i = i + 1
        # uncomment break in case you want to run only for one fold
        if i == 5:
            break


    test_images = get_image_names(images, test_set)


    test_x = images[test_set]
    test_y = masks[test_set]

    print(test_x)
    print(test_y)



    """ UNet Hyperparameters """
    H = 512
    W = 512
    size = (W, H)


    # loading the checkpoint file from the training process
    checκpoint_path = "files/2022_10_17_02_04_37_no_dropout_kfold_validation_fold_5.pth"


    """ Load the checkpoint file """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Loading the checkpoint 
    checkpoint = torch.load(checκpoint_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optim_state'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_losses']
    valid_loss = checkpoint['valid_losses']


    print(epoch)
    print(train_loss)
    print(valid_loss)

    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """Extract the name """
        name = x.split("/")[-1].split("\\")[-1].split(".")[0]
        print(name)

        ''' Reading image and mask '''
        image = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(y, cv2.IMREAD_COLOR)

        # Cropping the given image in order to have the right dimensions
        # the following code is actually used only on new images
        if image.shape != (3072, 3584):
            image = image[:, 128:3712]
            mask = mask[:, 128:3712]

        print("\n")
        #print(image.shape)
        #print(mask.shape)


        mask = rgb2mask(mask)
        #print(mask.shape)



        ''' Creating patches of the image and the corresponding mask'''
        patches_x = patchify(image, (512, 512), step=512)
        #print(patches_x.shape)
        patches_y = patchify(mask, (512, 512), step=512)
        #print(patches_y.shape)

        #pred_patches = np.empty(patches_y.shape)
        #print(pred_patches.shape)
        pred_patches = np.empty(patches_x.shape)
        #print(pred_patches.shape)


        for k in range(patches_x.shape[0]):
            for l in range(patches_x.shape[1]):

                x = np.expand_dims(patches_x[k, l, :, :], axis=0)                           # (1, 512, 512) 
                x = x/255.0
                x = np.expand_dims(x, axis=0)                           # (1, 1, 512, 512) to create a batch (egw mallon den to xreiazomai)
                #print("X_shape", x.shape)
                x = x.astype(np.float32)
                x = torch.from_numpy(x)
                x = x.to(device)


                y = np.expand_dims(patches_y[k, l, :, :], axis=0)            # (1, 512, 512)
                #y = np.expand_dims(y, axis=0)               # (1, 1, 512, 512) to create a batch
                y = torch.from_numpy(y)
                y = y.to(device)
                #print(y.shape)

                with torch.no_grad():
                    """ Prediction and Calculating FPS """
                    start_time = time.time()
                    pred_y = model(x)


                    # we need a softmax to get a mask as an output
                    pred_y = torch.softmax(pred_y, dim=1)
                    #print(pred_y.shape)
                    pred_y = torch.argmax(pred_y, dim=1)
                    #pred_y = np.expand_dims(pred_y, axis=0)
                    #print(pred_y.shape)


                    total_time = time.time() - start_time
                    time_taken.append(total_time)

                    #score = calculate_metrics(y, pred_y)
                    #metrics_score = list(map(add, metrics_score, score))
                    pred_y = pred_y.cpu().numpy()           ## (1, 512, 512)
                    #pred_y = pred_y * 255 // 2
                    pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
                    pred_y = np.array(pred_y, dtype=np.uint8)

                    ''' Saving the predicted patch to reconstruct the image at the next step '''
                    pred_patches[k, l, :, :] = pred_y


                """ Saving masks """
                # (512, 512) -> (3, 512, 512)
                image_3channel = mask_parse(patches_x[k, l, :, :])
                ori_mask = mask_parse(patches_y[k, l, :, :])
                pred_y = mask_parse(pred_patches[k, l, :, :])
                line = np.ones((size[1], 10, 3)) * 128

                #print(image_3channel.shape, ori_mask.shape, pred_y.shape)


                #print(np.unique(pred_y))

                cat_images = np.concatenate(
                    [image_3channel, line, ori_mask * 255 // 2, line, pred_y * 255 // 2], axis=1
                )

                # Creating an image where original image, ground truth and prediction 
                # are side by side to visualize and compare the results
                cv2.imwrite(f"results/{name}" + "_" + str(k) + "_" + str(l) +".png", cat_images)

        reconstructed_image = unpatchify(pred_patches * 255 // 2, image.shape)
        #reconstructed_image = mask_parse(reconstructed_image)
        #print(reconstructed_image.shape)
        cv2.imwrite(f"results_no_patches/{name}.png", reconstructed_image)
        #cv2.imwrite(f"results_no_patches/{name}.png", reconstructed_image)


        pred_mask = unpatchify(pred_patches, image.shape)


        score = calculate_metrics(mask, pred_mask)
        #print(score)

        #break