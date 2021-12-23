"""Implement Clip-Frame Constrast model."""
from mindspore.common import dtype as mstype
from backbones import VideoTransformerNetwork, ResNet3d
from mindspore import nn, ops
from mindspore import numpy as np
from mindspore import communication as dist
from ssvos.utils.param_utils import default_weight_init
from ssvos.utils.loss_utils import CrossEntropyLoss
from ssvos.utils.module_utils import Learnable_KSVD, NetWithSymmetricLoss


class CFC(nn.Cell):
    def __init__(self,
                 clip_branch=ResNet3d,
                 frame_branch=VideoTransformerNetwork,
                 dim=256,
                 mlp_dim=4096,
                 T=1.0,
                 clip_branch_cfg=dict(depth=50),
                 frame_branch_cfg=dict(seqlength=8, _2d_feat_extractor_depth=50)
                 ):
        super().__init__()
        self.T = T
        self.clip_branch = clip_branch(**clip_branch_cfg)
        self.frame_brach = frame_branch(**frame_branch_cfg) # we use resnet50 by default
        # make sure features from two branches have same dim
        assert self.clip_branch.feat_dim == self.frame_brach.feat_dim
        feat_dim = self.clip_branch.feat_dim
        # build projection mlp
        self.projector = self._build_mlp(2, feat_dim, mlp_dim, dim)
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

    def _init_weights(self):
        default_weight_init(self.projector)
        default_weight_init(self.predictor)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Dense(dim1, dim2, has_bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU())
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.SequentialCell(*mlp)

    def construct(self, x1, x2):
        # compute query features from two branches
        q1 = self.predictor(self.projector(self.frame_brach(x1)))
        q2 = self.predictor(self.projector(self.frame_brach(x2)))
        # compute key features from two branches
        k1 = self.predictor(self.projector(self.clip_branch(x1)))
        k2 = self.predictor(self.projector(self.clip_branch(x2)))

        return q1, q2, k1, k2


class CFCwithLearnableKSVD(CFC):
    def __init__(self,
                 clip_branch=ResNet3d,
                 frame_branch=VideoTransformerNetwork,
                 dim=256,
                 mlp_dim=4096,
                 T=1.,
                 dict_atoms=4096,
                 ISTA_iters=10):
        super().__init__(clip_branch=clip_branch,
                         frame_branch=frame_branch,
                         dim=dim,
                         mlp_dim=mlp_dim,
                         T=T)
        self.learnable_ksvd = Learnable_KSVD(
            dict_atoms=dict_atoms, ISTA_iters=ISTA_iters)
        self.cat = ops.Concat(axis=0)
        self.split = ops.Split(axis=0, output_num=4)
        self._init_weights()

    def _init_weights(self):
        super()._init_weights()
        default_weight_init(self.learnable_ksvd.lambda_predictor)

    def construct(self, x1, x2):
        q1, q2, k1, k2 = super().construct(x1, x2)
        x = self.cat((q1, q2, k1, k2))
        x = self.learnable_ksvd(x)

        return self.split(x)



if __name__=="__main__":
    # from mindspore import context
    # from mindspore.common.initializer import initializer, Normal
    # from ssvos.utils.dist_utils import init_dist

    # context.set_context(device_target='GPU', mode=context.PYNATIVE_MODE)
    # init_dist()

    # cfc = CFCwithLearnableKSVD()
    # cfc_InfoNCE = NetWithSymmetricLoss(cfc, InfoNCELoss())
    # dummy_input_1 = initializer(Normal(), (2,3,8,224,224))
    # dummy_input_2 = dummy_input_1.copy()
    # loss = cfc_InfoNCE(dummy_input_1, dummy_input_2)
    # print(loss)