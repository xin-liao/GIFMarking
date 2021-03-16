# -*- coding: utf-8 -*-
import gc
import inspect
import json
import csv
import os
from collections import Counter
import numpy as np
import imageio
import torch
from imageio import imread, imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from GIFGAN.utils_j import jpeg_compress_decompress
from torch.optim import Adam
from tqdm import tqdm
import random
import torch.nn as nn
from GIFGAN.ms_ssim_new import MSSSIM, SSIM
from GIFGAN.ms_ssim import MS_SSIM
from GIFGAN.utils import bits_to_bytearray, bytearray_to_text, ssim_frame, text_to_bits, vifp_mscale, ssim, \
    vifp_frame
from GIFGAN.vifp import vifp_mscale
from GIFGAN.noise import *
from GIFGAN.pytorch_cwt import *
from GIFGAN.pytorch_media_filter_3d import *
from GIFGAN.pytorch_salt_and_pepper_new import *
from GIFGAN.ContextualLoss import *

from GIFGAN.pqvi import *

from GIFGAN.gaussian import GaussianSmoothing
from torch.nn import functional as F
import sys
from tensorboardX import SummaryWriter
from GIFGAN.adv import Adv

sys.path.append(r'G:pjj\GIFGAN-master')
DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'train')

RES_LIST = [

    'val.encoder_mse',
    'val.decoder_loss',
    'val.ssim',
    'val.psnr',
    'val_sec.psnr',
    'val_sec.RMSE',
    'val_sec.ssim',
    'val_sec.vifb',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decode_ssim'

]

class GIFGAN(object):
    """
    Acknowledgement: Thanks to the open source code of SteganoGAN and StegaStamp for its contribution to our work. The reference paper is as following:

    Kevin Alex Zhang, Alfredo Cuesta-Infante, Lei Xu, Kalyan Veeramachaneni: SteganoGAN: High Capacity Image Steganography with GANs. CoRR abs/1901.03892 (2019)
    Matthew Tancik, Ben Mildenhall, Ren Ng: StegaStamp: Invisible Hyperlinks in Physical Photographs. CVPR 2020: 2114-2123
    """
    def _get_instance(self, class_or_instance, kwargs):
        """Returns an instance of the class"""

        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def set_to(self):
        self.device = torch.device('cuda')

    def set_device(self, cuda=True):
        """Sets the torch device depending on whether cuda is avaiable or not."""
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')



        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.pre.to(self.device)
        self.adv.to(self.device)
        # self.adv.to(self.device)
    def __init__(self, encoder, decoder, pre, adv, log_dir=None, **kwargs):
        self.encoder = self._get_instance(encoder, kwargs)
        self.decoder = self._get_instance(decoder, kwargs)
        self.pre = self._get_instance(pre, kwargs)
        self.adv =  self._get_instance(adv, kwargs)
        # self.adv.to(self.device)
        # self.adv = self._get_instance(adv, kwargs)
        self.set_device(cuda)
        # 对抗损失
        self.cover_label = 1
        self.encoded_label = 0
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.pre_optimizer = None
        self.decoder_optimizer = None
        self.fit_metrics = None
        self.history = list()

        self.log_dir = log_dir
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)



    def _encode_val(self, cover, secret, choose, quantize=False):
        """testing
        """
        pre = secret.unsqueeze(2)

        payload = self.pre(pre)
        # random_num = random.randint(0, 4)
        random_num = 5
        generated = self.encoder(cover, payload)

        if random_num == 0:

            # 帧删除（以补0）
            nums = [0, 1, 2, 3, 4, 5, 6, 7]
            x = random.randint(1, 3)

            y = random.randint(4, 7)
            zero_matrix1 = torch.zeros((1, 4, 1, 256, 256), device=self.device)

            generated1 = torch.chunk(generated, 8, dim=2)
            noised = generated1[nums[0]]
            for i in range(1, len(nums)):
                if i == x:
                    noised = torch.cat([noised] + [generated1[i]], dim=2)
                elif i == y:
                    noised = torch.cat([noised] + [generated1[i]], dim=2)
                else:
                    noised = torch.cat([noised] + [zero_matrix1], dim=2)
        if random_num == 1:
            # 帧置换
            nums = [0, 1, 2, 3, 4, 5, 6, 7]
            x = random.randint(0, 3)
            nums.append(x)
            z = random.randint(4, 7)
            nums.append(z)
            y = random.randint(4, 7)
            if y in nums:
                nums.remove(y)
            m = random.randint(0, 3)
            if m in nums:
                nums.remove(m)
            print(nums)
            generated1 = torch.chunk(generated, 8, dim=2)
            noised = generated1[nums[0]]
            for i in range(1, len(nums)):
                noised = torch.cat([noised] + [generated1[nums[i]]], dim=2)
        if random_num == 2:
            noised = median_blur3d(generated)
        # noised = generated
        if random_num == 3:
            noised = noise_saltand_pepper_3d(generated)
        if random_num == 4:
            # 高斯模糊
            smoothing = GaussianSmoothing(4, 5, 1)

            generated1 = F.pad(generated, (2, 2, 2, 2, 2, 2), mode='replicate')
            noised = smoothing(generated1)
        if random_num == 5:
            # JPEG压缩

            # print(cover_rgb_jpeg.shape, "jpeg")
            # exit()
            generated_j = generated.cpu()
            generated_j_chunk = torch.chunk(generated_j, 8, dim=2)

            for i in range(8):
                cover_temp = generated_j_chunk[i].squeeze(2)
                cover_rgba = torch.chunk(cover_temp, 4, dim=1)

                r = cover_rgba[0]
                g = cover_rgba[1]
                b = cover_rgba[2]

                cover_rgb = torch.cat([r] + [g] + [b], dim=1)

                cover_rgb_jpeg = jpeg_compress_decompress(cover_rgb)
                cover_rgb_jpeg = torch.cat([cover_rgb_jpeg]+[cover_rgba[3]], dim=1)
                cover_rgb_jpeg = cover_rgb_jpeg.unsqueeze(2)
                if i==0:
                    res = cover_rgb_jpeg
                else:

                    res = torch.cat([res]+[cover_rgb_jpeg], dim=2)
            noised = res.cuda()

        decoded = self.decoder(noised)

        return generated, noised, payload, decoded
    def _encode_decode(self, cover, secret, quantize=False):
    
        pre = secret.unsqueeze(2)

        payload = self.pre(pre)
        random_num = random.randint(0, 5)
        # random_num = 5
        generated = self.encoder(cover, payload)

        if random_num == 0:

            # 帧删除（以补0）
            nums = [0, 1, 2, 3, 4, 5, 6, 7]
            x = random.randint(1, 3)

            y = random.randint(4, 7)
            zero_matrix1 = torch.zeros((1, 4, 1, 256, 256), device=self.device)

            generated1 = torch.chunk(generated, 8, dim=2)
            noised = generated1[nums[0]]
            for i in range(1, len(nums)):
                if i == x:
                    noised = torch.cat([noised] + [generated1[i]], dim=2)
                elif i == y:
                    noised = torch.cat([noised] + [generated1[i]], dim=2)
                else:
                    noised = torch.cat([noised] + [zero_matrix1], dim=2)
        if random_num == 1:
            # 帧置换
            nums = [0, 1, 2, 3, 4, 5, 6, 7]
            x = random.randint(0, 3)
            nums.append(x)
            z = random.randint(4, 7)
            nums.append(z)
            y = random.randint(4, 7)
            if y in nums:
                nums.remove(y)
            m = random.randint(0, 3)
            if m in nums:
                nums.remove(m)
            print(nums)
            generated1 = torch.chunk(generated, 8, dim=2)
            noised = generated1[nums[0]]
            for i in range(1, len(nums)):
                noised = torch.cat([noised] + [generated1[nums[i]]], dim=2)
        if random_num == 2:
            noised = median_blur3d(generated)
        # noised = generated
        if random_num == 3:
            noised = noise_saltand_pepper_3d(generated)
        if random_num == 4:
            # 高斯模糊
            smoothing = GaussianSmoothing(4, 5, 1)

            generated1 = F.pad(generated, (2, 2, 2, 2, 2, 2), mode='replicate')
            noised = smoothing(generated1)
        if random_num == 5:
            # JPEG压缩

            # print(cover_rgb_jpeg.shape, "jpeg")
            # exit()
            generated_j = generated.cpu()
            generated_j_chunk = torch.chunk(generated_j, 8, dim=2)

            for i in range(8):
                cover_temp = generated_j_chunk[i].squeeze(2)
                cover_rgba = torch.chunk(cover_temp, 4, dim=1)

                r = cover_rgba[0]
                g = cover_rgba[1]
                b = cover_rgba[2]

                cover_rgb = torch.cat([r] + [g] + [b], dim=1)

                cover_rgb_jpeg = jpeg_compress_decompress(cover_rgb)
                cover_rgb_jpeg = torch.cat([cover_rgb_jpeg]+[cover_rgba[3]], dim=1)
                cover_rgb_jpeg = cover_rgb_jpeg.unsqueeze(2)
                if i==0:
                    res = cover_rgb_jpeg
                else:

                    res = torch.cat([res]+[cover_rgb_jpeg], dim=2)
            noised = res.cuda()


        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0
        # exit()
        decoded = self.decoder(noised)

        return generated, noised, payload, decoded

    def _pre(self, image):
        """Evaluate the image using the pre"""
        return torch.mean(self.adv(image))

    def _get_optimizers(self):
        _dec_list = list(self.encoder.parameters())+ list(self.decoder.parameters())+list(self.pre.parameters())
        decoder_optimizer = Adam(_dec_list, lr=1e-4)

        return decoder_optimizer
    def _get_optimizers1(self):
        _dec_list = list(self.adv.parameters())
        # pre_optimizer = Adam(self.pre.parameters(), lr=1e-4)
        adv_optimizer = Adam(_dec_list, lr=1e-4)

        return adv_optimizer
    # 对ADV进行训练
    def train_adv(self, train, secret_train, metrics):

        for cover, (secret, _) in tqdm(zip(train, secret_train), disable=self.verbose):
            gc.collect()

            cover = cover.to(self.device)
            secret = secret.to(self.device)
            pre = secret.unsqueeze(2)
            payload = self.pre(pre)
            # random_num = random.randint(0, 4)
            random_num = 0
            generated = self.encoder(cover, payload)
            cover_score = self._pre(cover)
            generated_score = self._pre(generated)
            self.adv_optimizer.zero_grad()
            # train on cover
            ((cover_score - generated_score)).backward(retain_graph=False)
            self.adv_optimizer.step()
        
    def train(self, train, secret_train, metrics):
        """Fit the encoder and the decoder on the train images."""
        i = 0
        # loss_fn = ps.PerceptualLoss()
        writer = SummaryWriter(log_dir='./log0907')
        layers = {
            "conv_1_1": 1.0,
            "conv_3_2": 1.0
        }
        #     I = torch.rand(1, 3, 128, 128).cuda()
        #     T = torch.randn(1, 3, 128, 128).cuda()
        # contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()
        for cover, (secret, _) in tqdm(zip(train, secret_train), disable=self.verbose):
            gc.collect()
            # cover_rgb_jpeg = jpeg_compress_decompress(secret)
            # print(secret)
            # exit()
            cover = cover.to(self.device)
            secret = secret.to(self.device)

            i += 1
            generated, noised, payload, decoded= self._encode_decode(cover, secret)
            encoder_mse, decoder_loss = self.compute_mse(cover, generated, secret, payload, decoded)
            
            ms_ssim_loss = MSSSIM()
            # payload = payload.narrow(2, 0, 1).squeeze(2)

            # ms_ssim = ms_ssim_loss(secret, decoded.narrow(2, 0, 1).squeeze(2))
            # encoder_LPIPS = torch.mean(loss_fn.forward(decoded.squeeze(2), secret))
            d_target_label_encoded = torch.full((1, 1), self.encoded_label, device=self.device)
            d_on_encoded = self.adv(generated)
            d_loss_on_encoded = binary_cross_entropy_with_logits(d_on_encoded, torch.zeros_like(d_on_encoded,device=self.device))
            # generated_score = d_loss_on_encoded
            # pqvi = pvqi(cover, generated).cuda()

            self.decoder_optimizer.zero_grad()

            # (100.0 * encoder_mse + 80 * (decoder_loss) + 1 * contex_loss(decoded.squeeze(2), secret)).backward()
            (100.0 * encoder_mse + 80 * (decoder_loss) +1*d_loss_on_encoded).backward()
            self.decoder_optimizer.step()
            ssim_loss = MS_SSIM(max_val=1)

            decode_ssim = ssim_loss(secret, decoded.squeeze(2))

            # writer.add_scalar('encoder_loss', encoder_mse.item(), i)
            # writer.add_scalar('decoder_loss', decoder_loss.item(), i)
            # writer.add_scalar('loss', (100.0 * encoder_mse + 80 * (decoder_loss)).item(), i)
            print('gen_loss', (100.0 * encoder_mse + 80 * (decoder_loss)+1*d_loss_on_encoded).item())
            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            # metrics['train.decoder_acc'].append(decoder_acc.item())
            metrics['train.decode_ssim'].append(decode_ssim.item())
        # writer.close()

   
    def compute_mse(self, cover, generated, secrect, payload, decoded):

        encoder_mse = mse_loss(generated, cover)


        decoder_loss = mse_loss(decoded.squeeze(2), secrect)

        return encoder_mse, decoder_loss

    
    def _validate(self, validate, secret_val, metrics):
        """Validation process"""
        for cover, (secret, _) in tqdm(zip(validate, secret_val), disable=self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            secret = secret.to(self.device)
            generated, noised, payload, decoded = self._encode_val(cover, secret, quantize=True)
            encoder_mse, decoder_loss = self.compute_mse(
                cover, generated, secret, payload, decoded)

            ssim_loss = MS_SSIM(max_val=1)
            decode_ssim = ssim_loss(secret, decoded.squeeze(2))

            print('val_psnr', 10 * torch.log10(4 / encoder_mse).item())
            # print('val.sec__8psnr', 10 * torch.log10(4 / mse_loss(decoded, payload)).item())
            print('val_sec.psnr', 10 * torch.log10(4 / mse_loss(decoded.squeeze(2), secret)).item())
            # print('noised_psnr',10 * torch.log10(4 / mse_loss(noised, generated)).item())
            # metrics['noised_psnr'].append(10 * torch.log10(4 / mse_loss(noised, generated)).item())
            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())

            metrics['val.ssim'].append(ssim_frame(cover, generated).item())
            metrics['val.psnr'].append(10 * torch.log10(4 / encoder_mse).item())
            # metrics['val.sec__8psnr'].append(10 * torch.log10(4 / mse_loss(decoded, payload)).item())
            metrics['val_sec.psnr'].append(
                10 * torch.log10(4 / mse_loss(decoded.squeeze(2), secret)).item())
            metrics['val_sec.RMSE'].append(torch.sqrt(mse_loss(decoded.squeeze(2), secret)).item())
            metrics['val_sec.ssim'].append(decode_ssim.item())

            # metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))
            metrics['val_sec.vifb'].append(vifp_mscale(((secret + 1) * 255).data.cpu().numpy(), (
                    (decoded.squeeze(2) + 1) * 255).data.cpu().numpy()))
    # 验证后进行采样，噪声图像是直接使用的椒盐噪声
    def sample_image(self, samples_path, cover, secrect, epoch, choose):
        cover = cover.to(self.device)
        secret = secrect.to(self.device)
        payload = self._secrect_gif(secret)
        generated = self.encoder(cover, payload)
        noised = self.pre(generated)
        decoded = self.decoder(noised)

        decoded = decoded.narrow(2, 0, 1).squeeze(2)
        samples = generated.size(0)
        for sample in range(samples):
            cover_path = os.path.join(samples_path, '{}.cover.gif'.format(sample))

            sample_name = '{}.generated-{:2d}.gif'.format(sample, epoch)
            sample_path = os.path.join(samples_path, sample_name)

            decode_name = '{}.decode-{:2d}.png'.format(sample, epoch)
            decoded_path = os.path.join(samples_path, decode_name)
            # sec原始图像存储
            sec_name = '{}.sec_cover-{:2d}.jpg'.format(sample, epoch)
            sec_path = os.path.join(samples_path, sec_name)

            payload_name = '{}.payload_cover-{:2d}.gif'.format(sample, epoch)
            payload_path = os.path.join(samples_path, payload_name)

            noise_name = '{}.noise-{:2d}.gif'.format(sample, epoch)
            noise_path = os.path.join(samples_path, noise_name)

            image = (cover[sample].permute(1, 2, 3, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.mimwrite(cover_path, (255.0 * image).astype('uint8'))

            sampled = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 3, 0)
            sampled = sampled.detach().cpu().numpy() + 1.0
            image = sampled / 2.0
            imageio.mimwrite(sample_path, (255.0 * image).astype('uint8'))

            payload = payload[sample].clamp(-1.0, 1.0).permute(1, 2, 3, 0)
            payload = payload.detach().cpu().numpy() + 1.0
            payload = payload / 2.0
            imageio.mimwrite(payload_path, (255.0 * payload).astype('uint8'))

            decoded = decoded[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            decoded = decoded.detach().cpu().numpy() + 1.0
            image = decoded / 2.0
            imageio.imwrite(decoded_path, (255.0 * image).astype('uint8'))

            sec_cover = (secrect[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(sec_path, (255.0 * sec_cover).astype('uint8'))

            noised = noised[sample].clamp(-1.0, 1.0).permute(1, 2, 3, 0)
            noised = noised.detach().cpu().numpy() + 1.0

            noised = noised / 2.0
            imageio.mimwrite(noise_path, (255.0 * noised).astype('uint8'))

    # 训练后进行采样，噪声图像是网络训练中
    def sample_image1(self, samples_path, cover, secrect, epoch, choose):
        print('generate_samples1')
        cover = cover.to(self.device)
        secrect = secrect.to(self.device)
        generated, noised, payload, decoded = self._encode_decode(cover, secrect, choose)
        decoded = decoded.squeeze(2)
        samples = generated.size(0)
        for sample in range(samples):
            cover_path = os.path.join(samples_path, '{}.cover—1.gif'.format(sample))

            sample_name = '{}.generated——1-{:2d}.gif'.format(sample, epoch)
            sample_path = os.path.join(samples_path, sample_name)

            decode_name = '{}.decode-1-{:2d}.jpg'.format(sample, epoch)
            decoded_path = os.path.join(samples_path, decode_name)
            # sec原始图像存储
            sec_name = '{}.sec_cover-1-{:2d}.jpg'.format(sample, epoch)
            sec_path = os.path.join(samples_path, sec_name)

            payload_name = '{}.payload_cover-1-{:2d}.gif'.format(sample, epoch)
            payload_path = os.path.join(samples_path, payload_name)

            noise_name = '{}.noise-1-{:2d}.gif'.format(sample, epoch)
            noise_path = os.path.join(samples_path, noise_name)

            image = (cover[sample].permute(1, 2, 3, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.mimwrite(cover_path, (255.0 * image).astype('uint8'))

            sampled = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 3, 0)
            sampled = sampled.detach().cpu().numpy() + 1.0
            image = sampled / 2.0
            imageio.mimwrite(sample_path, (255.0 * image).astype('uint8'))

            decoded = decoded[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            decoded = decoded.detach().cpu().numpy() + 1.0
            image = decoded / 2.0
            imageio.imwrite(decoded_path, (255.0 * image).astype('uint8'))

            sec_cover = (secrect[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(sec_path, (255.0 * sec_cover).astype('uint8'))

            noised = noised[sample].clamp(-1.0, 1.0).permute(1, 2, 3, 0)
            noised = noised.detach().cpu().numpy() + 1.0

            noised = noised / 2.0
            imageio.mimwrite(noise_path, (255.0 * noised).astype('uint8'))

   
    def fit(self, train, validate, secret_train, secret_val, epochs=1):
        """Train a new model with the given ImageLoader class."""
        # self.adv = BasicAdv()
        # self.adv.to(self.device)
        if self.adv_optimizer is None:
            self.adv_optimizer = self._get_optimizers1()

        if self.pre_optimizer is None:
            self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        if self.log_dir:
            sample_cover = next(iter(validate))
            sample_secret = next(iter(secret_val))[0]


        total = self.epochs + epochs
        for epoch in range(1, epochs + 1):
            self.epochs += 1

            metrics = {field: list() for field in RES_LIST}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))
            # self.sample_image(self.samples_path, sample_cover, epoch)
            # self.train_adv(train, secret_train, metrics, choose)
            # self.train_adv(train, secret_train, metrics, choose)
            self.train_adv(train, secret_train, metrics, choose)
            self.train(train, secret_train, metrics)
            save_name = '{}.bpp.p'.format(
                self.epochs)

            self.save(os.path.join(self.log_dir, save_name))
            self.sample_image1(self.samples_path, sample_cover, sample_secret, epoch, choose)
            self.fit_metrics = {k: sum(v) / 2500 for k, v in metrics.items()}

            self._validate(validate, secret_val, metrics, choose)

            self.fit_metrics = {k: sum(v) / 2500 for k, v in metrics.items()}
            f = open('G:\\pjj\\GIFGAN-master\\GIFGAN\\log_day_56\\vifb'+str(self.epochs)+'.csv', 'w', encoding='utf-8')
            csv_writer = csv.writer(f)
            print(1)
            for k, v in metrics.items():
                if k=='val_sec.vifb':
                    csv_writer.writerow(v)
            f.close()

            self.fit_metrics['epoch'] = epoch

            if self.log_dir:
                self.history.append(self.fit_metrics)

                metrics_path = os.path.join(self.log_dir, 'metrics.log')
                with open(metrics_path, 'a+') as metrics_file:
                    json.dump(self.history, metrics_file, indent=4)


                # self.sample_image1(self.samples_path, sample_cover, sample_secret, epoch)

            # Empty cuda cache (this may help for memory leaks)
            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()


    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, architecture=None, path=None, cuda=True, verbose=False):
        """Loads an instance of GIFGAN for the given architecture (default pretrained models)
        or loads a pretrained model from a given path.

        Args:
            architecture(str): Name of a pretrained model to be loaded from the default models.
            path(str): Path to custom pretrained model. *Architecture must be None.
            cuda(bool): Force loaded model to use cuda (if available).
            verbose(bool): Force loaded model to use or not verbose.
        """

        if architecture and not path:
            model_name = '{}.GM'.format(architecture)
            pretrained_path = os.path.join(os.path.dirname(__file__), 'pretrained')
            path = os.path.join(pretrained_path, model_name)

        elif (architecture is None and path is None) or (architecture and path):
            raise ValueError(
                'Please provide either an architecture or a path to pretrained model.')

        # GIFGAN = torch.load(path, map_location='cpu')
        GIFGAN = torch.load(path)
        print('xxxxx')
        GIFGAN.verbose = verbose

        GIFGAN.encoder.upgrade_legacy()
        GIFGAN.decoder.upgrade_legacy()
        GIFGAN.pre.upgrade_legacy()
        GIFGAN.adv.upgrade_legacy()
        # pre是噪声层
        GIFGAN.set_device(cuda)
        return GIFGAN
