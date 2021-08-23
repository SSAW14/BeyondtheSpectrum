# Beyond the Spectrum
Implementation for the IJCAI2021 work "Beyond the Spectrum: Detecting Deepfakes via Re-synthesis" by Yang He, Ning Yu, Margret Keuper and Mario Fritz.

## Pretrained Models
We release the model trained on CelebA-HQ dataset with image resolution 1024x1024. For the super resolution, we use 25,000 real images from CelebA-HQ to train it.
For the detectors, we use 25,000 real images and 25,000 fake images to train a binary classifier based on ResNet-50.

We release some models as examples to show how to apply our models based on pixel-level or stage5-level reconstruction errors to detect deepfakes.
Download link: https://drive.google.com/file/d/1FeIgABjBpjtnXT-Hl6p5a5lpZxINzXwv/view?usp=sharing.

If you have further questions regarding the trained models, please feel free to contact.

## Train
1. Train the super resolution model. 

We use Residual Dense Network (RDN) in our work. The following script shows the hyperparameters used in our experiments.
To be noticed, we only use 4 images to show how to run the script. For simplicity, you can download the pretrained model from the above link.

```bash
bash script/train_super_resolution_celeba.sh [GPU_ID]
```

2. Train the detectors.

After obtaining the super resolution, we use pixel-level or stage5-level L1 based recontruction error to train a classifier.
The following scripts use 10 training example to show how to train a classifier with a given super resolution model. You may need to adjust the learning rate and number of training epochs in your case.

```bash
bash script/train_pixel_pggan.sh [GPU_ID]
```

3. Finetune with auxiliary tasks

In order to improve the robustness of our detectors, we introduce auxiliary tasks (i.e., colorization or denoising) into the super resolution model training and finetune the whole model end-to-end.
The following scripts show how to train a model with those tasks.

```bash
bash script/train_pixel_pggan_colorization.sh [GPU_ID]
```
```bash
bash script/train_stage5_stylegan_denoising.sh [GPU_ID]
```

## Test
Please download our models. You can use pixel-level or stage5-level to perform deepfakes detection. 

```bash
bash script/test_pixel_celeba.sh [GPU_ID]
```
```bash
bash script/test_stage5_celeba.sh [GPU_ID]
```

## Citation
If our work is useful for you, please cite our paper:

    @inproceedings{yang_ijcai21,
      title={Beyond the Spectrum: Detecting Deepfakes via Re-synthesis},
      author={Yang He and Ning Yu and Margret Keuper and Mario Fritz},
      booktitle={30th International Joint Conference on Artificial Intelligence (IJCAI)},
      year={2021}
    }

Contact: Yang He (heyang614.cs@gmail.com)

Last update: 08-22-2021