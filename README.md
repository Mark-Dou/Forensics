# MIFAE-Forensics
This is the official implementation of MIFAE-Forensics for DeepFake detection.

## Catalog
- [x] Visualization demo.
- [x] Pre-training code.
- [ ] Pre-training checkpoints + Fine-tuning code.

## Network Structure.
![image](https://github.com/Mark-Dou/Forensics/blob/main/Visualization/MIFAE-Forensics.png)

Two pretext tasks, i.e. **facial region guided masking** in the spatial domain and **high-frequency components masking** in the frequency domain.
## Visualization Results.

### 1. Frequency Visualization
**Original image -> High-frequency components masking -> Network prediction -> Full reconstruction**

![image](https://github.com/Mark-Dou/Forensics/blob/main/Visualization/freq_recon.png)

### 2. Spatial Visualization
> + We first visualizae the MAE with facial region guiaded masking strategy in our paper.

**Original image -> Facial region guided masking -> Network prediction -> Full reconstruction**

![image](https://github.com/Mark-Dou/Forensics/blob/main/Visualization/spatial_guided.png)


> + We also visualize the **vanilla MAE** reconstruction **without** facial region guided masking strategy as comparison.

**Original image -> Random masking -> Network prediction -> Full reconstruction**

![image](https://github.com/Mark-Dou/Forensics/blob/main/Visualization/spatial_mae.png)

### 3. DeepFake detection via the reconstruction discrepancy.

![image](https://github.com/Mark-Dou/Forensics/blob/main/Visualization/recon.png)

## Usage

### Pre-training instruction
To pre-train ViT-B/16 (recommended default) with multi-node distributed training, run the following on 8 nodes with 8 GPUs each:

````
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --mask_radius 16 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
````

## Ackownledgement
This repository is built on [MAE](https://github.com/facebookresearch/mae/tree/main).




