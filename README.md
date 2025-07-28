# D2gFFT:Enhancing the targeted transferability of adversarial examples with dual domain gradient fusion fine-tuning
# Abstract

This repository contains the implementation of the **Dual-domain Gradient fusion fine-tuning Attack (D2gFFT)**, a Dual-Domain Gradient Fusion Fine-tuning (D2gFFT) method is proposed to enhance the transferability of adversarial samples. Specifically, the Discrete Cosine Transform(DCT) is used to transform the original and the adversarial samples from the spatial domain to the frequency domain, while the frequency perturbation strategy is designed to change the frequency characteristics of the samples and restore them to the spatial domain via the Inverse Discrete Cosine Transform(IDCT). Then, the gradient information in the spatial and frequency domains are fused and these fused gradients are used as a guide to fine-tune the feature layers of the model to generate the adversarial samples.

# Attack Process of D2gFFT

![https://github.com/Gitwanmeng/D2gFFT/blob/main/adv_imgs/lc.png](https://github.com/Gitwanmeng/D2gFFT/blob/main/adv_imgs/lc.png)

This figure shows the process of the D2gFFT method, which first obtains adversarial samples through baseline attacks, then masks the frequency and space separately to obtain gradients, fuses the dual domain gradients to obtain fused gradients, and finally fine tunes the feature space to generate more transferable adversarial samples

# Requiremens
To run this code,the following dependencies are required:

- Python 3.x
- PyTorch (>= 1.10.0)
- Numpy
- Torchvision
- pillow
- Matplotlib

# Usage

## Baseline Attack

To execute the baseline attack，use the following command:

`python Logit_Attack.py`

This will generate baseline adversarial samples（Logit）

## Fine-Tuning Attack

To execute the D2gFFT attack and verify the effectiveness of the attack, you can use the following command:

`python main_Feature_FT.py`

# Dateset

The 1000 images are from the NIPS 2017 ImageNet-Compatible dataset. [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) or [Zhao's Github](https://github.com/ZhengyuZhao/Targeted-Tansfer/tree/main/dataset).

# Experimental Setup

In our empirical evaluations, the D2gFFT method was tested on several popular architectures, including **Inception-v3**, **ResNet-50**, using the **ImageNet dataset**. The primary metric for evaluating attack performance was the **Attack Success Rate (ASR)**, which quantifies the percentage of adversarial examples that successfully mislead the target model.

