# MIFAE-Forensics
This is the official implementation of MIFAE-Forensics for DeepFake detection.

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
