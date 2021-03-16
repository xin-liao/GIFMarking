
from typing import Tuple
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_binary_kernel(window_size: Tuple[int, int]) -> torch.Tensor:
    r"""Creates a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    r"""Utility function that computes zero padding tuple."""
    computed: Tuple[int, ...] = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.MedianBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super(MedianBlur, self).__init__()
        self.kernel: torch.Tensor = _compute_binary_kernel(kernel_size)
        self.padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    def forward(self, input: torch.Tensor):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # map the local window to single vector
        features: torch.Tensor = F.conv2d(
            input, kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median



# functiona api


def median_blur(input: torch.Tensor,
                kernel_size: Tuple[int, int]) -> torch.Tensor:
    r"""Blurs an image using the median filter.

    See :class:`~kornia.filters.MedianBlur` for details.
    """
    return MedianBlur(kernel_size)(input)
def median_blur3d(img_torch):
    img_torch = img_torch.chunk(8, dim=2)
    img_torch_0 = img_torch[0].squeeze(2)
    img_torch_1 = img_torch[1].squeeze(2)
    img_torch_2 = img_torch[2].squeeze(2)
    img_torch_3 = img_torch[3].squeeze(2)
    img_torch_4 = img_torch[4].squeeze(2)
    img_torch_5 = img_torch[5].squeeze(2)
    img_torch_6 = img_torch[6].squeeze(2)
    img_torch_7 = img_torch[7].squeeze(2)

    img_torch_0 = median_blur(img_torch_0,(3,3)).unsqueeze(2)
    img_torch_1 = median_blur(img_torch_1,(3,3)).unsqueeze(2)
    img_torch_2 = median_blur(img_torch_2,(3,3)).unsqueeze(2)
    img_torch_3 = median_blur(img_torch_3,(3,3)).unsqueeze(2)
    img_torch_4 = median_blur(img_torch_4,(3,3)).unsqueeze(2)
    img_torch_5 = median_blur(img_torch_5,(3,3)).unsqueeze(2)
    img_torch_6 = median_blur(img_torch_6,(3,3)).unsqueeze(2)
    img_torch_7 = median_blur(img_torch_7,(3,3)).unsqueeze(2)

    img = torch.cat([img_torch_0] + [img_torch_1] + [img_torch_2] + [img_torch_3] + [img_torch_4] + [img_torch_5] + [
        img_torch_6] + [img_torch_7], dim=2)
    return img

# img = img.squeeze(0).permute(1,2,3,0)
# img = img.numpy().astype('uint8')
# imageio.mimsave("F:/pjj_ad/实验/samples/rotate.gif",img)
# print(img.shape)

# if __name__=="__main__":
#     img = imageio.mimread("F:/pjj_ad/实验/samples/0.cover.gif")
#     img = torch.Tensor(img).unsqueeze(0).permute(0,4,1,2,3)
#     print(img.shape)
#     img =  median_blur3d(img).squeeze(0).permute(1,2,3,0)
#     img  = img.numpy()
#     imageio.mimsave("F:/pjj_ad/实验/samples/m.gif",img.astype('uint8'))
