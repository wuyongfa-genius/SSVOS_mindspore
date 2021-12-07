"""Implement Clip-Frame Constrast model."""
from mindspore.common import dtype as mstype
from backbones import VideoTransformerNetwork, ResNet3d
from mindspore import nn, ops
from mindspore import numpy as np
from mindspore import communication as dist
from ssvos.utils.param_utils import default_weight_init
from ssvos.utils.loss_utils import CrossEntropyLoss
from ssvos.utils.module_utils import Learnable_KSVD


class CFC(nn.Cell):
    def __init__(self,
                 clip_branch=ResNet3d,
                 frame_branch=VideoTransformerNetwork,
                 dim=256,
                 mlp_dim=4096,
                 T=1.0
                 ):
        super().__init__()
        self.T = T
        self.clip_branch = clip_branch(depth=50)
        self.frame_brach = frame_branch(depth=50)
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

        return tuple(q1, q2, k1, k2)


class CFCwithLearnableKSVD(CFC):
    def __init__(self,
                 clip_branch=ResNet3d,
                 frame_branch=VideoTransformerNetwork,
                 dim=256,
                 mlp_dim=4096,
                 T=1,
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


class InfoNCELoss(nn.Cell):
    def __init__(self, T=0.3):
        super().__init__()
        self.T = T
        self.normalize = nn.Norm(axis=1)
        self.cross_entropy = CrossEntropyLoss()
        # init some ops to be used
        self.all_gather = ops.AllGather()
        self.matmul = ops.MatMul(transpose_b=True)

    def construct(self, q, k):
        N = q.shape[0]
        q = self.normalize(q)
        k = self.normalize(k)
        # gather all targets
        k = self.all_gather(k)
        logits = self.matmul(q, k) / self.T
        # create labels
        rank = dist.get_rank()
        labels = np.arange(N, dtype=mstype.int64) + N*rank
        cross_entropy = self.cross_entropy(logits, labels)

        return 2*self.T*cross_entropy


class NetWithSymmetricLoss(nn.Cell):
    """Wrap a Network with a symmetric Loss.
    Args:
        net(nn.Cell): A network with two branches outputing 
            (q1, q2, k1, k2)
        loss_fn(nn.Cell): A contrastive loss.
    """

    def __init__(self, net, loss_fn):
        super().__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn

    def construct(self, x1, x2):
        q1, q2, k1, k2 = self._net(x1, x2)

        return self._loss_fn(q1, k2) + self._loss_fn(q2, k1)
