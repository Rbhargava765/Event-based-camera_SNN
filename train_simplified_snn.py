import random
import sys
import os

# Add OF_EV_SNN-main to sys.path to allow imports from data, eval, etc.
current_script_dir = os.path.dirname(os.path.abspath(__file__))
of_ev_snn_main_dir = os.path.join(current_script_dir, 'OF_EV_SNN-main') # Path to OF_EV_SNN-main
sys.path.insert(0, of_ev_snn_main_dir) # Insert OF_EV_SNN-main itself for its modules

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as TvT

from spikingjelly.activation_based import functional
# from spikingjelly.clock_driven import neuron

from simplified_model import SimplifiedSNNModel

from tqdm import tqdm

from data.dsec_dataset_lite_stereo_21x9 import DSECDatasetLite 
from data.data_augmentation_2d import *

import numpy as np

from eval.vector_loss_functions import * 

# Enable GPU
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def set_random_seed(seed):
    #Python
    random.seed(seed)

    #Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # We remove deterministic implementations on pytorch due to the use of maximum pooling and replicate padding
    #if int(torch.__version__.split('.')[1]) < 8: # for pytorch < 1.8
    #   torch.set_deterministic(True)
    #else:
    #   torch.use_deterministic_algorithms(True)

    # NumPy
    np.random.seed(seed)

seed = 2305
set_random_seed(seed)

################################
## DATASET LOADING/GENERATION ##
################################

# Define desired temporal resolution (50ms between consecutive layers)
num_frames_per_ts = 11
forward_labels = 1

# Create training dataset
print("Creating Training Dataset ...")
# Construct absolute path for the root data directory
data_root_abs = os.path.join(of_ev_snn_main_dir, 'data', 'dataset', 'saved_flow_data')
expected_file_list_train = 'train_split_doubleseq.csv'
print(f"Expected full path for train_split_doubleseq.csv: {os.path.join(data_root_abs, 'sequence_lists', expected_file_list_train)}")
print(f"Passing absolute root='{data_root_abs}' to DSECDatasetLite for training.")
train_dataset = DSECDatasetLite(root=data_root_abs, file_list=expected_file_list_train, num_frames_per_ts=11, stereo=False, transform=None)

# Define training dataloader
batch_size = 1 # <-- ENSURING BATCH SIZE IS 1 FOR DRY RUN
batch_multiplyer = 1 # To artificially increase batch size, in case GPU memory were a constraint
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, pin_memory = True)


# Create validation dataset
print("Creating Validation Dataset ...")
# data_root_abs is already defined and is the same
expected_file_list_valid = 'valid_split_doubleseq.csv'
print(f"Expected full path for valid_split_doubleseq.csv: {os.path.join(data_root_abs, 'sequence_lists', expected_file_list_valid)}")
print(f"Passing absolute root='{data_root_abs}' to DSECDatasetLite for validation.")
valid_dataset = DSECDatasetLite(root=data_root_abs, file_list=expected_file_list_valid, num_frames_per_ts=11, stereo=False, transform=None)

# Define validation dataloader
valid_dataloader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 1, shuffle = False, drop_last = False, pin_memory = True)


########################
## TRAINING FRAMEWORK ##
########################

# Create the network

net = SimplifiedSNNModel(multiply_factor = 5.0).to(device)
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Trainable parameters: {}'.format(trainable_params))


# Initialize network weights

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


# Create the optimizer
lr = 2e-4
wd = 1e-2
optimizer = torch.optim.AdamW(net.parameters(), lr = lr, weight_decay = wd)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 20, 35], gamma = 0.5)
# Define the loss function

mod_fcn = mod_loss_function
lambda_mod = 1.

ang_fcn = angular_loss_function
lambda_ang = 1.

# Define the number of epochs

n_epochs = 1 # <-- MODIFIED FOR DRY RUN

###################
## TRAINING LOGS ##
###################

# Decide whether or not to store the network
save_net = True
test_acc = float('inf')

#####################
## SETUP FUNCTIONS ##
#####################

# Create data augmentation pipeline
data_augmentation = TvT.Compose([
#    Random_event_drop(),
#    Random_patch(p = 0.8),
    Random_horizontal_flip(p = 0.4),
#    Random_vertical_flip(p = 0.15),
#    Random_rotate(p = 0.1), # ONLY IF WORKING WITH SQUARE TENSORS
])

##########################
## TRAIN, EVAL AND TEST ##
##########################

n_chunks_train = len(train_dataloader)
n_chunks_valid = len(valid_dataloader) 

for epoch in range(n_epochs):

    print(f'Epoch {epoch}')

    net.train()
    
    running_loss = 0.

    epoch_mod_loss = 0.
    epoch_ang_loss = 0.

    batch_iter = 0

    print('Training...')
    for chunk, mask, label in tqdm(train_dataloader):

        functional.reset_net(net)

        chunk = torch.transpose(chunk, 1, 2)

        mask = torch.unsqueeze(mask, dim = 1)
        
        label = label.to(device = device, dtype = torch.float32) # [Batch, 2, H_orig, W_orig]
        chunk = chunk.to(device = device, dtype = torch.float32)
        mask = mask.to(device = device)

        if chunk.shape[2] > 11:
            start_t = (chunk.shape[2] - 11) // 2
            chunk = chunk[:, :, start_t:start_t+11, :, :]

        B, C, T, H_orig, W_orig = chunk.shape
        chunk_reshaped = chunk.reshape(B * C * T, 1, H_orig, W_orig)
        chunk_resized_reshaped = F.interpolate(chunk_reshaped, size=(256, 256), mode='bilinear', align_corners=False)
        chunk = chunk_resized_reshaped.reshape(B, C, T, 256, 256)

        label = F.interpolate(label, size=(256, 256), mode='bilinear', align_corners=False)
        if H_orig != 256 or W_orig != 256:
             label[:, 0, :, :] *= (256.0 / W_orig)
             label[:, 1, :, :] *= (256.0 / H_orig)

        mask = F.interpolate(mask.float(), size=(256, 256), mode='nearest').byte()

        chunk, label, mask = data_augmentation([chunk, label, mask])

        pred_list = net(chunk)
        
        pred = pred_list
        mod_loss = mod_fcn(pred, label, mask)
        ang_loss = ang_fcn(pred, label, mask)
        curr_loss = lambda_mod * mod_loss + lambda_ang * ang_loss
        
        if np.isnan(curr_loss.item()):
            raise

        curr_loss.backward()

        batch_iter += 1
        if batch_iter % batch_multiplyer == 0:
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5, norm_type=2)
            optimizer.step()

            optimizer.zero_grad()

        
        running_loss += curr_loss.item() * batch_size

        epoch_mod_loss += mod_loss.item() * batch_size
        epoch_ang_loss += ang_loss.item() * batch_size

        break # <-- ADDED FOR DRY RUN (Process only one training batch)

    epoch_loss = running_loss / n_chunks_train

    epoch_mod_loss /= n_chunks_train
    epoch_ang_loss /= n_chunks_train
    
    print(f'Epoch loss = {epoch_loss}')


    # Training Dataset (eval)

    net.eval()

    epoch_mod_loss = 0.
    epoch_ang_loss = 0.

    print('Validating... (training sequence)')

    for chunk, mask, label in tqdm(train_dataloader):

        functional.reset_net(net)
        
        chunk = torch.transpose(chunk, 1, 2)

        mask = torch.unsqueeze(mask, dim = 1)

        chunk = chunk.to(device = device, dtype = torch.float32)
        label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]
        mask = mask.to(device = device)
        
        with torch.no_grad():
            pred = net(chunk)

        mod_loss = mod_fcn(pred, label, mask)
        ang_loss = ang_fcn(pred, label, mask)

        epoch_mod_loss += mod_loss.item() * batch_size
        epoch_ang_loss += ang_loss.item() * batch_size
        break # <-- ADDED FOR DRY RUN (Process only one validation batch on train_data)


    epoch_mod_loss /= n_chunks_train
    epoch_ang_loss /= n_chunks_train

    epoch_loss_train_eval = epoch_mod_loss + epoch_ang_loss
    print('Epoch loss (Validation): {} \n'.format(epoch_loss_train_eval))



    # Validation Dataset

    pred_sequence = []
    label_sequence = []

    net.eval()

    epoch_mod_loss_test = 0.
    epoch_ang_loss_test = 0.

    print('Validating... (test sequence)')

    for chunk, mask, label in tqdm(valid_dataloader):

        functional.reset_net(net)

        chunk = torch.transpose(chunk, 1, 2)

        mask = torch.unsqueeze(mask, dim = 1)
        
        chunk = chunk.to(device = device, dtype = torch.float32)
        label = label.to(device = device, dtype = torch.float32) # [num_batches, 2, H, W]
        mask = mask.to(device = device)

        with torch.no_grad():
            pred = net(chunk)

        
        mod_loss = mod_fcn(pred, label, mask)
        ang_loss = ang_fcn(pred, label, mask)

        epoch_mod_loss_test += mod_loss.item() * batch_size
        epoch_ang_loss_test += ang_loss.item() * batch_size
        

        pred_sequence.append(torch.squeeze(pred[0,:,:,:]).cpu().detach().numpy())
        label_sequence.append(torch.squeeze(label[0,:,:,:]).cpu().detach().numpy())
        break # <-- ADDED FOR DRY RUN (Process only one validation batch on valid_data)
    

    epoch_mod_loss_test /= n_chunks_valid
    epoch_ang_loss_test /= n_chunks_valid

    epoch_loss_valid = epoch_mod_loss_test + epoch_ang_loss_test
    print('Epoch loss (Validation): {} \n'.format(epoch_loss_valid))
    
    # Save the network
    # We only save the network if it beats the previous best result (mod loss) on the validation set
    if save_net & (epoch_loss_valid < test_acc):       

        test_acc = epoch_loss_valid
        torch.save(net.state_dict(), 'results/checkpoint_epoch{}.pth'.format(epoch))


    scheduler.step()

print('SO FAR, EVERYTHING IS WORKING!!!')
