# Data-Free-Generative-Model-Stealing
## Folder 1-5: Basic Attacking Results 
Use MNIST Dataset
* Scenario 1 cGAN -> cDCGAN
* Scenario 2 cDCGAN -> cGAN
* Scenario 3 cDCGAN -> cDCGAN
* Scenario 4 cDiffusion -> cDCGAN  
  The code of cDiffusion is modified from: https://github.com/TeaPearce/Conditional_Diffusion_MNIST/tree/main
* Scenario 5 cDCGAN -> cDiffusion

## Folder 6: Improving with Augmentations
Based on Scenario 1
* Basic Transforms
* Differentiable Augmentation  
  The code is from the original repo of DiffAug: https://github.com/mit-han-lab/data-efficient-gans
* Mixup Enhancement  
  The code is modified from the original repo of Mixup: https://github.com/hongyi-zhang/mixup

## Folder 7-8: Additional Experiment
Use Anime Avatar Dataset from: https://www.kaggle.com/datasets/splcher/animefacedataset?resource=download
* cGAN -> cDCGAN
* cDCGAN -> cGAN
