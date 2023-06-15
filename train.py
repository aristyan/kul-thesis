import time
import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader

from utils import epoch_time, seeding, create_dir, seed_worker, _init_fn

from model import build_unet
from model2 import unet
import random

from loss import DiceLoss, DiceBCELoss, CELoss
from data import CustomDataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pandas as pd
import wandb

from pytorchtools import EarlyStopping
from focal_loss.focal_loss import FocalLoss

""" Training function to calculate the training epoch loss """
def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    # indicating that the model is training
    # (important for dropout and batch_norm layers)
    model.train()

    # going through all the batches in the loader
    for x, y in loader:
        # loading the images and the corresponding masks to the GPU device
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # setting the gradients to zero
        optimizer.zero_grad()

        # making a prediction
        y_pred = model(x)

        # finding the loss between the prediction and the ground truth
        loss = loss_fn(y_pred, y)

        # performing back propagation
        loss.backward()

        # optimizing the loss function using the optimizer
        optimizer.step()

        torch.cuda.empty_cache()

        # adding each batch loss to calculate later the average of all batches
        epoch_loss += loss.item()

    # calculating the mean loss
    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


""" Validation function to calculate the validation epoch loss """
def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    # turn off dropout, batch norm, etc. layers
    # during model evaluation
    model.eval()

    # turning off gradients computation
    with torch.no_grad():
        # going through all the batches in the loader
        for x, y in loader:
            # loading the images and the corresponding masks to the GPU device
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # making a prediction
            y_pred = model(x)

            # finding the loss between the prediction and the ground truth
            loss = loss_fn(y_pred, y)

            # adding each batch loss to calculate later the average of all batches
            epoch_loss += loss.item()

        # calculating the mean loss
        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":

    # Login to wandb account
    wandb.login()

    # Set a configuration dictionary to save the parameters
    config = {
        "learning_rate": "1e-5",
        "batch_size": 16,
        "epochs": 100,
        "loss": "CE",
        "dropout": "True",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "early_stopping": "True",
        "class_weights": "True"
    }


    # Init wandb run (from now on wandb tracks your output, saved parameters, etc.)
    wandb.init(
        # Set the project where this run will be logged
        project="unet_256_dropout_class_weights_fold_1",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        # name=f"experiment_{run}",
        # Track hyperparameters and run metadata
        config=config)

    print(config)

    # creating the directories in case they do not exist
    # create_dir("/esat/biomeddata/guests/r0877855/files/")
    # create_dir("/esat/biomeddata/guests/r0877855/losses/")

    """ Seeding """
    seeding(42)

    """ Loading the original images and masks """
    path = "/esat/biomeddata/guests/r0877855/dataset"
    #path = "../dataset"

    images = np.array(sorted(glob(os.path.join(path, "images", "*.tif"))))
    masks = np.array(sorted(glob(os.path.join(path, "masks_rgb", "*.png"))))
    print("----------------------------------------------------------")
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
        # and choose your split/fold
        if i == 1:
            break

    train_images = images[train_set]
    valid_images = images[valid_set]
    test_images = images[test_set]

    train_masks = masks[train_set]
    valid_masks = masks[valid_set]
    test_masks = masks[test_set]
    
    print("----------------------------------------------------------")
    print("Train Images:\n", train_images)
    print("Validation Images:\n", valid_images)
    print("Test Images:\n", test_images)


    """ Data Augmentation """

    # Create transforms using albumentations library

    ###########################   SOS   ###########################
    # Many non-spatial transformations like CLAHE, RandomBrightness,
    # RandomContrast, RandomGamma can be also added. They will be
    # applied only to the image and not the mask.
    # https://albumentations.ai/docs/examples/example_kaggle_salt/

    # in our case GaussNoise and Normalize will only be applied to image

    train_transform = A.Compose([
        A.Rotate(limit=45, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, always_apply=False, p=0.5),
        A.OpticalDistortion(p=0.5),
        A.GridDistortion(p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.Normalize(
            mean=0.0,
            std=1.0,
            max_pixel_value=255.0
        ),
        ToTensorV2()],
        # in case you want to apply the same transform on interest mask
        # additional_targets={'mask0': 'mask'}
    )

    valid_transform = A.Compose([
        A.Normalize(
            mean=0.0,
            std=1.0,
            max_pixel_value=255.0
        ),
        ToTensorV2()],
        # in case you want to apply the same transform on interest mask
        # additional_targets={'mask0': 'mask'}
    )

    test_transform = A.Compose([
        A.Normalize(
            mean=0.0,
            std=1.0,
            max_pixel_value=255.0
        ),
        ToTensorV2()],
        # in case you want to apply the same transform on interest mask
        # additional_targets={'mask0': 'mask'}
    )

    train_dataset = CustomDataset(train_images, train_masks, patch_size=256, split='train', transform=train_transform)
    valid_dataset = CustomDataset(valid_images, valid_masks, patch_size=256, split='valid', transform=valid_transform)
    test_dataset = CustomDataset(test_images, test_masks, patch_size=256, split='test', transform=test_transform)


    #print(train_dataset[0][0][0])

    """ Hyperparameters """
    
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    lr = float(config["learning_rate"])

    # print("batch_size:", batch_size)
    # print("num_epochs:", num_epochs)
    # print("lr:", lr)

    #g = torch.Generator()
    #g.manual_seed(0)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        worker_init_fn=_init_fn,
        #generator=g
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        worker_init_fn=_init_fn,
        #generator=g
    )

    # Sanity check
    # printing train_loader length
    # it has to be len(train_dataset)/batch_size

    #print(len(train_loader))
    #print(len(train_dataset) / batch_size)

    #print(len(valid_loader))
    #print(len(valid_dataset) / batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device used:", device)

    # build_unet channels: 64 -> 128 -> 256 -> 512 -> 1024
    # model = build_unet()
    # model = model.to(device)

    # unet channels: 32 -> 64 -> 128 -> 256 -> 512
    model = unet()
    model = model.to(device)

    # Uncomment to view model architecture
    #print("Model Architecture:\n\n", model)

    #for param in model.parameters():
    #    print(param)
    #    break


    # Binary Cross Entropy -> Binary Semantic Segmentation
    # loss_fn = DiceBCELoss()

    # Categorical Cross Entropy -> Binary Semantic Segmentation
    # loss_fn = DiceLoss()

    # Categorical Cross Entropy -> Multiclass Semantic Segmentation
    # loss_fn = CELoss()

    # Dealing with the imbalanced dataset - using class weights
    loss_fn = CELoss(weight=torch.tensor([0.4879760710608399, 1.5724569302121914, 3.176905469512336]).cuda())

    # Focal loss package
    #!pip install focal_loss_torch

    # Withoout class weights
    # loss_fn = FocalLoss(gamma=0.7)

    # with weights
    # The weights parameter is similar to the alpha value mentioned in the paper
    # weights = torch.FloatTensor([2, 3.2, 0.7])
    # loss_fn = FocalLoss(gamma=0.7, weights=weights)

    # to ignore index
    # loss_fn = FocalLoss(gamma=0.7, ignore_index=0)

    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # , weight_decay=0.1)

    # adjusting the learning rate based on the number of epochs by using a scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, verbose=True)

    # early stopping patience; how long to wait after last time validation loss improved.
    patience = 15

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # possible future format to save the different experiments
    #{timestamp}_loss_{loss}_lr_{lr}_weights_{class_weights}_dropout_{dropout}_opt_{optimizer}_batch_size_{batch_size}_earlyStop_{early_stopping}_scheduler_{scheduler}_reduction_none_{reduction_none}_fold_{i}
    create_dir("./files/")
    create_dir("./losses/")

    # defining the checkpoint path to save the model weights
    timestr = time.strftime("%Y_%m_%d_%H_%M_%S_")

    str = f"downscale_image_dropout_class_weights_batch_size_{batch_size}_fold_{i}_reduced_channels.pth"

    #checkpoint_path = "/esat/biomeddata/guests/r0877855/files/" + timestr + str
    checkpoint_path = "files/" + timestr + str
    print("Checkpoint path:", checkpoint_path)

    # defining a loss file to save the train and validation loss through time
    #loss_file_name = f"/esat/biomeddata/guests/r0877855/losses/{timestr}no_dropout_lr_{lr}_batch_size_{batch_size}_fold_{i}_reduced_channels.csv"
    loss_file_name = f"losses/{timestr}downscale_image_dropout_class_weights_lr_{lr}_batch_size_{batch_size}_fold_{i}_reduced_channels.csv"
    print("Loss file name:", loss_file_name)

    """ Training the model """
    # Used as a checkpoint
    best_valid_loss = float("inf")

    train_losses = []
    valid_losses = []

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=10)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        # Tell wandb to logg epochs, train loss and valid loss
        wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}"
            print(data_str)

            best_valid_loss = valid_loss

            # Just to save the model
            # torch.save(model.state_dict(), checkpoint_path)

            # Creating a checkpoint to save the best epoch
            # model state dict (weights & biases), optimizer state,
            # training losses and validation losses for printing purpose

            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "train_losses": train_losses,
                "valid_losses": valid_losses
            }

            torch.save(checkpoint, checkpoint_path)

        # Scheduler to reduce lr if not improved after a number of epochs
        scheduler.step(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.3f}\n"
        data_str += f"\tVal. Loss: {valid_loss:.3f}\n"
        print(data_str)

        # Keep track of the train and validation loss
        # in order to plot them later
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # dictionary of lists
        loss_dict = {'train_loss': train_losses, 'val_loss': valid_losses}

        df = pd.DataFrame(loss_dict)

        # saving the dataframe
        df.to_csv(loss_file_name)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

    wandb.finish()