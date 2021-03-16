import torch
import torch.nn as nn
import numpy as np
import imageio

def noise_saltand_pepper_2d(img1):
    '''
    2D图像的椒盐噪声
    :param img1:1*3*256*256
    :return: 椒盐噪声image
    '''

    salt = torch.zeros_like(img1)
    pepper = torch.ones_like(img1)
    ratio = int(img1.shape[2]*img1.shape[3]*3)
    mask_1 = np.random.choice([0.0, 1.0], ratio, p=[0.1,0.9 ])
    # 组成0，255的shape与image一样的随机矩阵
    mask_0_255 = np.random.choice([0.0, 255.0], ratio, p=[0.5, 0.5])
    # 0.1的概率为0，其余为0
    mask_1_tensor = torch.tensor(mask_1,  dtype=torch.float).reshape(1,3,256,256)
    # 用来放噪声的点
    mask_pepper = pepper - mask_1_tensor

    mask_0_255_tensor = torch.tensor(mask_0_255, dtype=torch.float).reshape(1, 3, 256, 256)
    noise = mask_0_255_tensor*mask_pepper
    # 丢失部分数据
    img1 = img1*mask_1_tensor + noise


    return img1
        # mask_tensor.unsqueeze_(0)
        # mask_tensor.unsqueeze_(0)
    # mask_tensor = mask_tensor.expand_as(img1)

def noise_saltand_pepper_3d(img_tensor):
    '''
    3D图像的椒盐噪声
    :param img1:1*4 *8*256*256
    :return: 椒盐噪声image
    '''

    pepper = torch.ones_like(img_tensor,device='cuda')
    ratio = int(img_tensor.shape[3] * img_tensor.shape[4] * 4 * 8)
    # 0.05的椒盐噪声
    mask_1 = np.random.choice([0.0, 1.0], ratio, p=[0.005, 0.995])
    # 组成0，255的shape与image一样的随机矩阵
    #因为图像已经归一化（-1,1），所以直接加0,255是不对的，应该是-1,1
    mask_0_255 = np.random.choice([-1.0, 1.0], ratio, p=[0.5, 0.5])
    # 0.1的概率为0，其余为1
    mask_1_tensor = torch.tensor(mask_1, dtype=torch.float).reshape(1, 4, 8, 256, 256).cuda()
    # 用来放噪声的点
    mask_pepper = pepper - mask_1_tensor
    mask_0_255_tensor = torch.tensor(mask_0_255, dtype=torch.float).reshape(1, 4, 8, 256, 256).cuda()
    noise = mask_0_255_tensor * mask_pepper
    # 丢失部分数据
    img1 = img_tensor * mask_1_tensor + noise
    return img1
# if __name__ =='__main__':
#     img = imageio.mimread('F:/pjj_ad/实验/samples/0.cover.gif')
#     img_tensor = torch.Tensor(img).permute(3,0,1,2).unsqueeze(0)
#     pepper = torch.ones_like(img_tensor)
#     ratio = int(img_tensor.shape[3] * img_tensor.shape[4] * 4*8)
#     mask_1 = np.random.choice([0.0, 1.0], ratio, p=[0.1, 0.9])
#     # 组成0，255的shape与image一样的随机矩阵
#     mask_0_255 = np.random.choice([0.0, 255.0], ratio, p=[0.5, 0.5])
#     # 0.1的概率为0，其余为1
#     mask_1_tensor = torch.tensor(mask_1, dtype=torch.float).reshape(1, 4, 8, 256, 256)
#     # 用来放噪声的点
#     mask_pepper = pepper - mask_1_tensor
#
#     mask_0_255_tensor = torch.tensor(mask_0_255, dtype=torch.float).reshape(1, 4, 8, 256, 256)
#     noise = mask_0_255_tensor * mask_pepper
#
#     # 丢失部分数据
#     img1 = img_tensor * mask_1_tensor + noise
#     img1 = img1.squeeze(0).permute(1,2,3,0)
#     img1 = img1.detach().cpu().numpy()
#     imageio.mimsave('F:/pjj_ad/实验/samples/0.salt.gif',img1.astype('uint8'))
