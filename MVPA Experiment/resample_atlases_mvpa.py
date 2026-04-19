"""
Resample atlases to each group's functional resolution.
========================================================
Run this ONCE before the volumetric MVPA pipeline. It:
  1. Fetches the Destrieux and Harvard-Oxford atlases from nilearn
  2. Resamples each to match your live- and studio-group functional grids
  3. Writes them to disk so the main pipeline can load them directly

For each group, pick any cleaned functional file as the reference —
all participants in a group should share the same voxel grid, so one
reference is enough.
"""

from pathlib import Path
from nilearn.datasets import fetch_atlas_destrieux_2009, fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img
import nibabel as nib
import os


# === CONFIGURE ===============================================================
# Point these at ANY cleaned functional .nii.gz from each group.
bids_dir = "/home/NEU480/datasets/narratives/"
cleaned_dir = bids_dir + "derivatives/afni-nosmooth/"
net_id = os.environ['USER']
scratch_folder = f"/scratch/network/{net_id}/"
thesis_folder = scratch_folder + "thesis/"

LIVE_REFERENCE   = cleaned_dir + "sub-020/func/sub-020_task-pieman_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz"
STUDIO_REFERENCE = cleaned_dir + "sub-300/func/sub-300_task-piemanpni_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz"

OUTPUT_DIR = Path(thesis_folder + "atlases")
# =============================================================================


def resample_one(atlas_img, reference_path, out_path):
    resampled = resample_to_img(
        source_img=atlas_img,
        target_img=reference_path,
        interpolation="nearest",   # REQUIRED for label images
        force_resample=True,
        copy_header=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_filename(str(out_path))

    # Quick sanity check
    ref_shape = nib.load(reference_path).shape[:3]
    assert resampled.shape == ref_shape, (
        f"Resampling mismatch: got {resampled.shape}, expected {ref_shape}"
    )
    print(f"  -> {out_path}  shape={resampled.shape}")


def main():
    print("Fetching atlases from nilearn...")
    destrieux = fetch_atlas_destrieux_2009()
    harvard_oxford = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    harvard_oxford_sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    # destrieux.maps is a path; harvard_oxford.maps is a NIfTI image
    destrieux_img = destrieux.maps
    ho_img = harvard_oxford.maps
    hos_img = harvard_oxford_sub.maps

    for group, ref in [("live", LIVE_REFERENCE), ("studio", STUDIO_REFERENCE)]:
        print(f"\n[{group}] reference: {ref}")
        print(f"[{group}] reference shape: {nib.load(ref).shape}")

        resample_one(
            destrieux_img,
            ref,
            OUTPUT_DIR / group / "destrieux_resampled.nii.gz",
        )
        resample_one(
            ho_img,
            ref,
            OUTPUT_DIR / group / "harvard_oxford_resampled.nii.gz",
        )
        resample_one(
            hos_img,
            ref,
            OUTPUT_DIR / group / "harvard_oxford_sub_resampled.nii.gz",
        )
    print("\nDone. Use these paths in your MVPA pipeline config:")
    print(f'  LIVE_ATLAS_PATHS = {{')
    print(f'      "destrieux":      "{OUTPUT_DIR}/live/destrieux_resampled.nii.gz",')
    print(f'      "harvard_oxford": "{OUTPUT_DIR}/live/harvard_oxford_resampled.nii.gz",')
    print(f'  }}')
    print(f'  STUDIO_ATLAS_PATHS = {{')
    print(f'      "destrieux":      "{OUTPUT_DIR}/studio/destrieux_resampled.nii.gz",')
    print(f'      "harvard_oxford": "{OUTPUT_DIR}/studio/harvard_oxford_resampled.nii.gz",')
    print(f'  }}')


if __name__ == "__main__":
    main()