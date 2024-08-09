# Lit-Net
**Harnessing Multi-resolution and Multi-scale Attention for Underwater Image Restoration.**
- Currently, the paper is submitted to [**The Visual Computer**](https://link.springer.com/journal/371).

![Block](LitNet_dia.png)

- This paper deals with the **underwater image restoration**. 
- For this, we have considered two of the main low-level vision tasks, 
  - **Underwater image enhancement**,
  - **Underwater image super-resolution**.

## Dataset
- For underwater image enhancement 
  - [**EUVP**](http://irvlab.cs.umn.edu/resources/euvp-dataset), 
  - [**UIEB**](https://li-chongyi.github.io/proj_benchmark.html), and
  - [**SUIM-E**](https://drive.google.com/drive/folders/1gA3Ic7yOSbHd3w214-AgMI9UleAt4bRM).
- For super-resolution,
    - [**UFO-120**](http://irvlab.cs.umn.edu/resources/ufo-120-dataset). 

Rrequirements as given below.
```
Python 3.5.2
Pytorch '1.0.1.post2'
torchvision 0.2.2
opencv 4.0.0
scipy 1.2.1
numpy 1.16.2
tqdm
```

### Training
- Use the below command for training:
```
python train.py --checkpoints_dir --batch_size --learning_rate             
```
### Testing
- Use the below command for testing:
```
python test.py  
```
### For Underwater Semantic Segmentation
- To generate segmentation maps on enhanced images, follow [**SUIM**](https://github.com/xahidbuffon/SUIM). 

### Send us feedback
- If you have any queries or feedback, please contact us @(**p.alik@iitg.ac.in**).

 <!-- ### Acknowledgements -->
<!-- - Some portion of the code are adapted from [**DeepWaveNet**](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration). The authors greatfully acknowledge it! -->
