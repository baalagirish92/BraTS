

import torchio as tio
import torch
import multiprocessing
import random

from Unet import UNet

def get_patch_loader(type, dataset, batch_size,patch_size, samples_per_volume,max_queue_length ):
    sampler = tio.data.UniformSampler(patch_size)
    num_workers = multiprocessing.cpu_count()
    if type=="train":
        patches_set = tio.Queue(
        subjects_dataset=dataset,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
        )

    else:
        patches_set = tio.Queue(
        subjects_dataset=dataset,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
        )
    patch_loader = torch.utils.data.DataLoader(patches_set, batch_size=batch_size)
    
    return patch_loader


def get_model(in_channels, n_classes, depth=3, wf=6, padding=True,batch_norm=False, up_mode='upconv', dropout=False): 
    model= UNet(in_channels, n_classes, depth, wf, padding,batch_norm, up_mode, dropout)
    return model

def test_visualize(test_set, model, batch_size,device):
    subject = random.choice(test_set)
    patch_size = 48, 48, 48  # we can user larger patches for inference
    patch_overlap = 4, 4, 4
    grid_sampler = tio.inference.GridSampler(
        subject,
        patch_size,
        patch_overlap,
    )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")

    model.eval()
    with torch.no_grad():
        for patches_batch in patch_loader:
            inputs = patches_batch['mri'][tio.DATA].to(device, dtype=torch.float)
            locations = patches_batch[tio.LOCATION]
            labels = model(inputs).argmax(dim=1, keepdim=True)
            aggregator.add_batch(labels, locations)

    foreground = aggregator.get_output_tensor()
    affine = subject.mri.affine
    prediction = tio.ScalarImage(tensor=foreground, affine=affine)
    subject.add_image(prediction, 'prediction')
    subject.plot(figsize=(9, 8), cmap_dict={'prediction': 'cubehelix'})