import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F

from utils_Frequency import *
from utils_ImageNetCom import Normalize, FIAloss, DI, DI_keepresolution, gkern
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

class FeatureD2gFFT(object):
    # imageNet
    def __init__(self,p, model=None, device=None, epsilon=16 / 255., k=10, alpha=1.6 / 255., prob=0.7,
                 mask_num=30, mu=1.0, model_name='res18',beta=0.2,rho=0.5,k_wu = 5,decay_factor = 0.8):
        # set Parameters
        self.model = model.to(device)
        self.epsilon = epsilon
        self.k = k + k_wu
        self.k_wu = k_wu
        self.alpha = alpha
        self.prob = prob  # for normal model, drop 0.3; for defense models, drop 0.1
        self.mask_num = mask_num  # according to paper 30
        self.mu = mu
        self.device = device
        self.model_name = model_name
        self.beta = beta    # 初始化
        self.rho = rho      # 初始化
        self.decay_factor = decay_factor
        self.p = p

    def frequency_enhancement(self,x, beta=0.2, rho=0.5):

        gauss = torch.randn_like(x) * beta * self.epsilon
        x_noisy = torch.clamp(x + gauss, 0, 1)


        x_dct = dct_2d(x_noisy)


        mask = torch.ones_like(x_dct)
        h, w = x_dct.shape[-2:]
        mask[:, :, h // 4:, w // 4:] = rho  


        x_idct = idct_2d(x_dct * mask)
        return x_idct
    def perturb(self, X_nat, X_adv, y_tar, y_ori):
        self.alpha = self.epsilon / 16.
        # get grads
        labels_tar = y_tar.clone().detach().to(self.device)
        labels_ori = y_ori.clone().detach().to(self.device)
        # in place
        _, temp_x_l1, temp_x_l2, temp_x_l3, temp_x_l4 = self.model.features_grad_multi_layers(X_nat)

        batch_size = X_nat.shape[0]
        image_size = X_nat.shape[-1]

        # calculate the feature importance (to y_o) from the clean image
        grad_sum_l1 = torch.zeros(temp_x_l1.shape).to(self.device)
        grad_sum_l2 = torch.zeros(temp_x_l2.shape).to(self.device)
        grad_sum_l3 = torch.zeros(temp_x_l3.shape).to(self.device)
        grad_sum_l4 = torch.zeros(temp_x_l4.shape).to(self.device)

        # Feature importance for clean image
        for i in range(self.mask_num):
            self.model.zero_grad()
            img_temp_i = norm(X_nat).clone()
            mask = torch.tensor(np.random.binomial(1, self.prob, size=(batch_size, 3, image_size, image_size))).to(self.device)
            img_temp_i = img_temp_i * mask
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            logit_label = logits.gather(1, labels_ori.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()
            grad_sum_l1 += x_l1.grad
            grad_sum_l2 += x_l2.grad
            grad_sum_l3 += x_l3.grad
            grad_sum_l4 += x_l4.grad

        grad_sum_l1 = grad_sum_l1 / grad_sum_l1.std()
        grad_sum_l2 = grad_sum_l2 / grad_sum_l2.std()
        grad_sum_l3 = grad_sum_l3 / grad_sum_l3.std()
        grad_sum_l4 = grad_sum_l4 / grad_sum_l4.std()

        # Feature importance for adversarial image
        grad_sum_mid_l1 = torch.zeros(temp_x_l1.shape).to(self.device)
        grad_sum_mid_l2 = torch.zeros(temp_x_l2.shape).to(self.device)
        grad_sum_mid_l3 = torch.zeros(temp_x_l3.shape).to(self.device)
        grad_sum_mid_l4 = torch.zeros(temp_x_l4.shape).to(self.device)

        for i in range(self.mask_num):
            self.model.zero_grad()
            img_temp_i = norm(X_adv).clone()
            mask = torch.tensor(np.random.binomial(1, self.prob, size=(batch_size, 3, image_size, image_size))).to(self.device)
            img_temp_i = img_temp_i * mask
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            logit_label = logits.gather(1, labels_tar.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()
            grad_sum_mid_l1 += x_l1.grad
            grad_sum_mid_l2 += x_l2.grad
            grad_sum_mid_l3 += x_l3.grad
            grad_sum_mid_l4 += x_l4.grad

        grad_sum_mid_l1 = grad_sum_mid_l1 / grad_sum_mid_l1.std()
        grad_sum_mid_l2 = grad_sum_mid_l2 / grad_sum_mid_l2.std()
        grad_sum_mid_l3 = grad_sum_mid_l3 / grad_sum_mid_l3.std()
        grad_sum_mid_l4 = grad_sum_mid_l4 / grad_sum_mid_l4.std()

        # Feature importance for clean image after DCT and IDCT
        grad_sum_dct_l1 = torch.zeros(temp_x_l1.shape).to(self.device)
        grad_sum_dct_l2 = torch.zeros(temp_x_l2.shape).to(self.device)
        grad_sum_dct_l3 = torch.zeros(temp_x_l3.shape).to(self.device)
        grad_sum_dct_l4 = torch.zeros(temp_x_l4.shape).to(self.device)

        # Apply DCT, add noise and mask, then IDCT for clean image
        # x_nat_idct = self.frequency_enhancement(X_nat);
        x_nat_idct = dct_2d(X_nat).cuda()

        for i in range(self.mask_num):
            self.model.zero_grad()
            # gauss = torch.randn(batch_size, 3, 299, 299) * (16 / 255)
            # gauss = gauss.cuda()
            # x_nat_idct = dct_2d(X_nat + gauss).cuda()
            img_temp_i = norm(x_nat_idct).clone()
            # mask = torch.tensor(np.random.binomial(1, self.prob, size=(batch_size, 3, image_size, image_size))).to(self.device)
            mask = (torch.rand_like(X_nat) * 2 * self.rho + 1 - self.rho).cuda()
            img_temp_i = idct_2d(img_temp_i * mask)
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            logit_label = logits.gather(1, labels_ori.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()
            grad_sum_dct_l1 += x_l1.grad
            grad_sum_dct_l2 += x_l2.grad
            grad_sum_dct_l3 += x_l3.grad
            grad_sum_dct_l4 += x_l4.grad

        grad_sum_dct_l1 = grad_sum_dct_l1 / grad_sum_dct_l1.std()
        grad_sum_dct_l2 = grad_sum_dct_l2 / grad_sum_dct_l2.std()
        grad_sum_dct_l3 = grad_sum_dct_l3 / grad_sum_dct_l3.std()
        grad_sum_dct_l4 = grad_sum_dct_l4 / grad_sum_dct_l4.std()

        # Feature importance for adversarial image after DCT and IDCT
        grad_sum_mid_dct_l1 = torch.zeros(temp_x_l1.shape).to(self.device)
        grad_sum_mid_dct_l2 = torch.zeros(temp_x_l2.shape).to(self.device)
        grad_sum_mid_dct_l3 = torch.zeros(temp_x_l3.shape).to(self.device)
        grad_sum_mid_dct_l4 = torch.zeros(temp_x_l4.shape).to(self.device)

        # Apply DCT, add noise and mask, then IDCT for adversarial image
        # x_adv_idct = self.frequency_enhancement(X_nat);
        x_adv_idct = dct_2d(X_adv).cuda()
        for i in range(self.mask_num):
            self.model.zero_grad()
            # gauss = torch.randn(batch_size, 3, 299, 299) * (16 / 255)
            # gauss = gauss.cuda()
            # x_adv_idct = dct_2d(X_adv + gauss).cuda()
            img_temp_i = norm(x_adv_idct).clone()
            # mask = torch.tensor(np.random.binomial(1, self.prob, size=(batch_size, 3, image_size, image_size))).to(self.device)
            mask = (torch.rand_like(img_temp_i) * 2 * self.rho + 1 - self.rho).cuda()
            img_temp_i = idct_2d(img_temp_i * mask)
            logits, x_l1, x_l2, x_l3, x_l4 = self.model.features_grad_multi_layers(img_temp_i)
            logit_label = logits.gather(1, labels_tar.unsqueeze(1)).squeeze(1)
            logit_label.sum().backward()
            grad_sum_mid_dct_l1 += x_l1.grad
            grad_sum_mid_dct_l2 += x_l2.grad
            grad_sum_mid_dct_l3 += x_l3.grad
            grad_sum_mid_dct_l4 += x_l4.grad

        grad_sum_mid_dct_l1 = grad_sum_mid_dct_l1 / grad_sum_mid_dct_l1.std()
        grad_sum_mid_dct_l2 = grad_sum_mid_dct_l2 / grad_sum_mid_dct_l2.std()
        grad_sum_mid_dct_l3 = grad_sum_mid_dct_l3 / grad_sum_mid_dct_l3.std()
        grad_sum_mid_dct_l4 = grad_sum_mid_dct_l4 / grad_sum_mid_dct_l4.std()

        # Combine gradients
        beta = 0.2
        # 计算原始对抗样本和DCT变换后对抗样本的梯度平均



        # grad_sum_new_l1 = (grad_sum_mid_l1 * (1 - self.p) + grad_sum_mid_dct_l1 * self.p) - beta * (grad_sum_l1 * (1 - self.p) + grad_sum_dct_l1 * self.p)
        # grad_sum_new_l2 = (grad_sum_mid_l2 * (1 - self.p) + grad_sum_mid_dct_l2 * self.p) - beta * (grad_sum_l2 * (1 - self.p) + grad_sum_dct_l2 * self.p)
        grad_sum_new_l3 = (grad_sum_mid_l3 * (1 - self.p) + grad_sum_mid_dct_l3 * self.p) - beta * (grad_sum_l3 * (1 - self.p) + grad_sum_dct_l3 * self.p)
        # grad_sum_new_l4 = (grad_sum_mid_l4 * (1 - self.p) + grad_sum_mid_dct_l4 * self.p) - beta * (grad_sum_l4 * (1 - self.p) + grad_sum_dct_l4 * self.p)

        # initialization
        g = 0
        x_cle = X_nat.detach()
        x_adv_2 = X_adv.clone()
        eta_prev = 0
        accumaleted_factor = 0

        for epoch in range(self.k):
            self.model.zero_grad()
            x_adv_2.requires_grad_()
            x_adv_2_DI = DI_keepresolution(x_adv_2)  # DI
            x_adv_norm = norm(x_adv_2_DI)  # [0, 1] to [-1, 1]

            _, _, mid_feature_l3, _ = self.model.multi_layer_features(x_adv_norm)
            loss = FIAloss(grad_sum_new_l3, mid_feature_l3)  # FIA loss
            loss.backward()

            grad_c = x_adv_2.grad
            # TI, MI
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
            g = self.mu * g + grad_c

            # update AE
            x_adv_2 = x_adv_2 + self.alpha * g.sign()
            with torch.no_grad():
                eta = x_adv_2 - x_cle

                ## Core code of AaF
                # memory-efficient implementaion of:
                # x_0*decay_factor^(epoch) + x_1*decay_factor^(epoch-1) + ...+ x_(epoch)
                if epoch >= self.k_wu:         # 5 (10 for CE) iterations for warming up
                    eta = (eta + accumaleted_factor * eta_prev) / (accumaleted_factor + 1)
                    accumaleted_factor = self.decay_factor * (accumaleted_factor + 1)
                    eta_prev = eta

                eta = torch.clamp(eta, min=-self.epsilon, max=self.epsilon)
                X_adv_AaF = torch.clamp(x_cle + eta, min=0, max=1).detach_()

            x_adv_2 = torch.clamp(x_cle + eta, min=0, max=1)

        return X_adv_AaF

# patch_by_strides() is borrowed from RPA paper
def patch_by_strides(img_shape, patch_size, prob):
    X_mask = np.ones(img_shape)
    N0, H0, W0, C0 = X_mask.shape
    ph = H0 // patch_size[0]
    pw = W0 // patch_size[1]
    X = X_mask[:, :ph * patch_size[0], :pw * patch_size[1]]
    N, H, W, C = X.shape
    shape = (N, ph, pw, patch_size[0], patch_size[1], C)
    strides = (X.strides[0], X.strides[1] * patch_size[0], X.strides[2] * patch_size[0], *X.strides[1:])
    mask_patchs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    mask_len = mask_patchs.shape[1] * mask_patchs.shape[2] * mask_patchs.shape[-1]
    ran_num = int(mask_len * (1 - prob))
    rand_list = np.random.choice(mask_len, ran_num, replace=False)
    for i in range(mask_patchs.shape[1]):
        for j in range(mask_patchs.shape[2]):
            for k in range(mask_patchs.shape[-1]):
                if i * mask_patchs.shape[2] * mask_patchs.shape[-1] + j * mask_patchs.shape[-1] + k in rand_list:
                    mask_patchs[:, i, j, :, :, k] = np.random.uniform(0, 1,
                                                                      (N, mask_patchs.shape[3], mask_patchs.shape[4]))
    img2 = np.concatenate(mask_patchs, axis=0, )
    img2 = np.concatenate(img2, axis=1)
    img2 = np.concatenate(img2, axis=1)
    img2 = img2.reshape((N, H, W, C))
    X_mask[:, :ph * patch_size[0], :pw * patch_size[1]] = img2
    return X_mask

