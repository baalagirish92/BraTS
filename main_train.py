# Uses ideas from Torchio tutorials

import torch
import torchio as tio
import random 
import multiprocessing
import matplotlib.pyplot as plt
from BrainTumorData import BrainTumorData
from utils import get_patch_loader,get_model,test_visualize
from torch.utils.tensorboard import SummaryWriter
import statistics
import monai
from tqdm import tqdm
# Config
seed = 42 

random.seed(seed)
torch.manual_seed(seed)


num_workers = multiprocessing.cpu_count()
plt.rcParams['figure.figsize'] = 12, 6


image_folder="G:\BraTS2021_Training_Data"
image_type="t1ce"

dataset=BrainTumorData(image_folder,image_type, (1000,51,200))

dataset.prepare_data()
dataset.setup()

training_batch_size = 5
validation_batch_size =5 
num_workers = multiprocessing.cpu_count()

patch_size = 32
samples_per_volume = 20
max_queue_length = 120
sampler = tio.data.UniformSampler(patch_size)

training_loader_patches = get_patch_loader("train", dataset.train_set, training_batch_size, patch_size, samples_per_volume, max_queue_length)

validation_loader_patches=get_patch_loader("valid", dataset.val_set, validation_batch_size, patch_size, samples_per_volume, max_queue_length)


writer = SummaryWriter()

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model = get_model(in_channels=1, n_classes=4, depth=3, wf=6, padding=True,batch_norm=False, up_mode='upconv', dropout=False).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0001)

Epochs=10
criterion= monai.losses.DiceCELoss(sigmoid=True)


#Validation Loop
def validation_loop():
    print(("validating......."))
    overall_validation_loss = []
    model.eval()
    for batch in validation_loader_patches:
        inputs = batch['mri'][tio.DATA].to(device, dtype=torch.float)
        targets = batch['brain'][tio.DATA].to(device, dtype=torch.float)
        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits, targets)
        overall_validation_loss.append(loss.item())
    model.train()
    validation_loss = statistics.mean(overall_validation_loss)
    return validation_loss


#Training loop
steps = 0
old_validation_loss = 0
for epoch in range(Epochs):
    overall_training_loss = []
    opt.zero_grad()
    for i,batch in enumerate(tqdm(training_loader_patches)):
        steps += 1
        inputs = batch['mri'][tio.DATA].to(device, dtype=torch.float)
        targets = batch['brain'][tio.DATA].to(device, dtype=torch.float)
        
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        overall_training_loss.append(loss.item())
        loss.backward()
        
        opt.zero_grad()
        opt.step()
    #Train Loss and validation loss seggregation
    training_loss = statistics.mean(overall_training_loss)
    writer.add_scalar("Loss/train", training_loss, epoch)
    #test_network(epoch)
    validation_loss = validation_loop()
    writer.add_scalar("Loss/validation", validation_loss, epoch)
    print(f"epoch {epoch} : training_loss ===> {training_loss} || Validation_loss ===> {validation_loss} \n")
    if (old_validation_loss == 0) or (old_validation_loss > validation_loss):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss}, 'BraTs2021_10epochs_1000T_51V_DiceCE_loss_v1' + ".pth")
        print("model_saved")
    old_validation_loss = validation_loss

test_visualize(dataset.test_set, model, 5, device)