"""Some module utils in MindSpore."""
from mindspore import nn, Parameter, ops
from mindspore import numpy as np
from mindspore.common.initializer import TruncatedNormal, initializer


class Identity(nn.Cell):
    def construct(self, x):
        return x


class MLP(nn.Cell):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 bias=True,
                 norm_layer=nn.BatchNorm1d,
                 act_layer=nn.ReLU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        keep_drop_probs = (1-drop, 1-drop)

        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=bias)
        self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(keep_drop_probs[0])
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop2 = nn.Dropout(keep_drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvModule(nn.Cell):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=1,
                 stride=1,
                 pad_mode='pad',
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        if conv_cfg['type'] == 'Conv2d':
            if isinstance(padding, int):
                mindspore_padding = padding
            elif isinstance(padding, tuple) and len(padding) == 2:
                mindspore_padding = (
                    padding[0], padding[0], padding[1], padding[1])
            self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride, pad_mode, mindspore_padding,
                                  dilation, groups, bias)
        elif conv_cfg['type'] == 'Conv3d':
            if isinstance(padding, int):
                mindspore_padding = padding
            elif isinstance(padding, tuple) and len(padding) == 3:
                mindspore_padding = (
                    padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
            self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, pad_mode, mindspore_padding,
                                  dilation, groups, bias)
        if norm_cfg is not None:
            if norm_cfg['type'] == 'BN2d':
                self.norm = nn.BatchNorm2d(planes)
            elif norm_cfg['type'] == 'BN3d':
                self.norm = nn.BatchNorm3d(planes)
        else:
            self.norm = None

        if act_cfg is not None:
            if act_cfg['type'] == 'ReLU':
                self.act = nn.ReLU()
            else:
                raise NotImplementedError
        else:
            self.act = None

    def construct(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Learnable_KSVD(nn.Cell):
    def __init__(self,
                 feat_dim=256,
                 dict_atoms=4096,
                 ISTA_iters=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.dict_atoms = dict_atoms
        self.ISTA_iters = ISTA_iters
        self.lambda_predictor = nn.SequentialCell(
            nn.Dense(feat_dim, feat_dim//4, has_bias=False),
            nn.BatchNorm1d(feat_dim//4),
            nn.ReLU(),
            nn.Dense(feat_dim//4, feat_dim//16, has_bias=False),
            nn.BatchNorm1d(feat_dim//16),
            nn.ReLU(),
            nn.Dense(feat_dim//16, 1)
        )
        dictionary, squared_spectral_norm = self._init_dict()
        self.dictionary = Parameter(dictionary)  # d*N
        self.c = Parameter(squared_spectral_norm)
        self.e = Parameter(np.eye(dictionary.shape[-1]), name='Identity')
        # init ops to be used
        self.matmul = ops.MatMul()
        self.sign = ops.Sign()
        self.relu = ops.ReLU()
        self.abs = ops.Abs()

    def _init_dict(self):
        dictionary = initializer(TruncatedNormal(
            sigma=.02), (self.feat_dim, self.dict_atoms))  # d*N
        # normalize
        dictionary = np.norm(dictionary, axis=1)
        # compute the spectral norm of dict
        spectral_norm = np.norm(dictionary, ord=2)

        return dictionary, spectral_norm**2

    def _soft_shrink(self, x, thresh):
        return self.sign(x)*self.relu(self.abs(x)-thresh)

    def construct(self, x):
        # normalize dict before every call
        dictionary = np.norm(self.dictionary, axis=1)

        lam = self.lambda_predictor(x)
        thresh = lam/self.c
        # Iterative Soft-Thresholding Shrinkage Iteration
        M1 = self.e-1/self.c * \
            self.matmul(dictionary.transpose(), dictionary)  # N*N
        M2 = 1/self.c * \
            self.matmul(dictionary.transpose(), x.transpose())  # N*B
        alpha = self._soft_shrink(M2, thresh)
        for _ in range(self.ISTA_iters):
            alpha = self._soft_shrink(self.matmul(M1, alpha)+M2, thresh)
        # reconstruct x from alpha and dict
        rec_x = self.matmul(dictionary, alpha).transpose()

        return rec_x
