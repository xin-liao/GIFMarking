
import numpy as np
from torch.nn import functional as F
import math
import torch
import imageio

# 噪声层：椒盐噪声
def salt_and_pepper(X, prop):
	X_clone=X.view(-1, 1)
    # print(X_clone.shape)
	num_feature=X_clone.size(0)
	mn=X_clone.min()
	mx=X_clone.max()
	indices=np.random.randint(0, num_feature, int(num_feature*prop))
#	print indices
	for elem in indices :
		if np.random.random() < 0.5 :
			X_clone[elem]=mn
		else :
			X_clone[elem]=mx
	return X_clone.view(X.size())
