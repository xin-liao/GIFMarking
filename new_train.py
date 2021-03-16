from GIFMarking import GIFGAN
from GIFMarking.noise_train import noise
from GIFMarking.decoder_hide import Decoder
from GIFMarking.encoders_hide import Encoder
from GIFMarking.pre import Pre_network
from GIFMarking.adv import Adv
from GIFMarking.noise_train import noise
from GIFMarking.loader import DataLoader,DataLoader1
import torchvision
from torchvision import transforms
import torch


# DE
def train():
    """
    Acknowledgement: Thanks to the open source code of SteganoGAN and StegaStamp for its contribution to our work. The reference paper is

    Kevin Alex Zhang, Alfredo Cuesta-Infante, Lei Xu, Kalyan Veeramachaneni: SteganoGAN: High Capacity Image Steganography with GANs. CoRR abs/1901.03892 (2019)
    Matthew Tancik, Ben Mildenhall, Ren Ng: StegaStamp: Invisible Hyperlinks in Physical Photographs. CVPR 2020: 2114-2123
    """
    #train dataset and test dataset
    train = DataLoader('H:/Tgif/data/train_last/', batch_size=1,shuffle=True)
    validation = DataLoader('H:/Tgif/data/val_last/',batch_size=1,shuffle=True)

    secret_train = DataLoader1('H:/LOGO/train_fin/', batch_size=1)
    secret_val = DataLoader1('H:/LOGO/test_fin/', batch_size=1)

    GIFMarking = GIFGAN(Encoder, Decoder, Pre_network, Adv, cuda=True, verbose=False, log_dir='hide')
    # load the trained model
    # GIFMarking = GIFGAN.load('G:/pjj/GIFGAN-master/GIFMarking/log_day_56/adv1-logo')
    GIFMarking.fit(train, validate, secret_train, secret_val, epochs=50)



if __name__ == '__main__':
    train()
