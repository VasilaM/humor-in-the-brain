from nilearn import datasets, image
import nibabel as nib
import numpy as np
import os

os.makedirs("data/masks", exist_ok=True)

# Fetch atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", data_dir=dir_nilearn)
atlas_img = nib.load(atlas.maps)
atlas_data = atlas.maps.get_fdata()
labels = atlas.labels

for i, label in enumerate(labels):
    print(i, label)