# Multi Scale GroupNorm GAN for High Fidelity CT Image Synthesis

## Introduction

Computed Tomography (CT) is central to modern diagnosis, but high resolution scans require higher radiation dose, while low dose scans often look noisy and lose fine anatomical detail. This project explores a deep learning approach for synthesizing realistic 3D CT style brain volumes that preserve anatomical structure without increasing radiation exposure.  

We build on Generative Adversarial Networks (GANs) and adapt them to volumetric data, focusing on stability, memory efficiency, and the ability to capture both global brain shape and local details. The model is trained on preprocessed 3D brain volumes from the Brain Genomics Superstruct Project (GSP) dataset.

## Overview

- **Task**  
  Generate high fidelity 3D CT like brain volumes that are anatomically consistent with real scans and suitable for data augmentation and downstream analysis.

- **Core idea**  
  Combine multi scale generation with Group Normalization to handle large 3D volumes under tight GPU memory and small batch size constraints.

- **Architecture at a glance**  
  - A hierarchical generator that first learns coarse global structure at low resolution, then refines local anatomy at higher resolution.  
  - Dual scale discriminators that check both overall brain shape and local patch level realism.  
  - An encoder pathway used to reconstruct real volumes, which stabilizes training and encourages voxel wise fidelity.

- **Data and preprocessing**  
  - 3D brain volumes from the GSP dataset.  
  - Standard steps such as skull stripping, registration to a common space, resampling to a fixed volume size, and intensity normalization to a consistent range.

## Methodology

1. **Data pipeline**  
   Preprocessed GSP brain volumes are loaded as 3D tensors, normalized, and optionally augmented with random flips, rotations, and crops. The pipeline produces pairs of full volumes and high resolution sub volumes that are used jointly during training.

2. **Multi scale generator**  
   - A latent vector is mapped to a shared 3D feature representation.  
   - A low resolution branch produces a coarse global volume that captures overall brain geometry.  
   - A high resolution branch takes features from the shared representation and focuses on sub volumes, learning fine structures such as cortical folds and small subcortical regions.  
   - At inference time the model uses what it learned from patches to generate an entire high resolution volume in one forward pass, avoiding explicit patch stitching.

3. **Encoder and reconstruction path**  
   - An encoder processes real low resolution volumes and high resolution sub volumes into feature spaces aligned with the generator.  
   - The generator then reconstructs these inputs, and a voxel wise reconstruction loss (such as L1) is applied.  
   - This reconstruction objective works alongside the adversarial loss to keep the generated volumes structurally close to real CT style data and to reduce mode collapse.

4. **Dual discriminators with GroupNorm**  
   - A global discriminator judges downsampled full volumes, enforcing realistic overall anatomy.  
   - A local discriminator focuses on randomly sampled high resolution patches, enforcing sharp textures and local consistency.  
   - Both use Group Normalization in 3D convolutional blocks so that training remains stable even when batch size is very small.

5. **Training loop**  
   - For each batch, the generator produces synthetic volumes and patches, the encoder reconstructs real data, and both discriminators are updated to distinguish real from fake.  
   - The generator and encoder are trained to minimize a combination of adversarial loss and reconstruction loss, while the discriminators maximize their classification accuracy.  
   - Checkpoints and visualizations of synthesized volumes are saved regularly to track convergence and qualitative behavior.

This methodology results in a memory efficient 3D GAN framework that can synthesize anatomically coherent, high resolution CT style brain volumes while remaining practical to train on standard GPU hardware.
