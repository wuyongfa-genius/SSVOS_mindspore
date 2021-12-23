"""Implement BYOL model in MindSpore."""
import copy
from mindspore import nn, ops
from mindspore.ops import stop_gradient
from ssvos.utils.module_utils import NetWithSymmetricLoss
from ssvos.utils.loss_utils import CosineSimilarityLoss


class BYOL(nn.Cell):
    def __init__(self, encoder: nn.Cell, hidden_dim=4096, out_dim=256, momentum=0.996):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.momentum = momentum
        feat_dim = getattr(encoder, 'feat_dim', 2048)

        projection_mlp = self._build_mlp(feat_dim, hidden_dim, out_dim)

        self.online_encoder = nn.SequentialCell([
            encoder,
            projection_mlp,
        ])
        self.momentum_encoder = copy.deepcopy(self.online_encoder)
        # set momentum's param to requires no grad
        for params in self.momentum_encoder.trainable_params():
            params.requires_grad = False

        self.prediction_mlp = self._build_mlp(out_dim, hidden_dim, out_dim)

    @staticmethod
    def _build_mlp(in_dim, hidden_dim, out_dim):
        return nn.SequentialCell([
            nn.Dense(in_dim, hidden_dim, has_bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dense(hidden_dim, out_dim)
        ])

    @staticmethod
    def _get_momentum_updating_params(cell:nn.Cell):
        # exclude bn.moving_mean and bn.moving_variance
        return list(filter(lambda x: 'moving_' not in x.name, cell.untrainable_params()))

    def _update_momentum_params(self):
        for online_params, momentum_params in zip(self.online_encoder.trainable_params(),
                                                  self._get_momentum_updating_params(self.momentum_encoder)):
            old_params, up_params = momentum_params.data, online_params.data
            momentum_params.set_data(self.momentum * \
                old_params + (1-self.momentum)*up_params)

    def construct(self, x1, x2):
        online_pred1 = self.prediction_mlp(self.online_encoder(x1))
        online_pred2 = self.prediction_mlp(self.online_encoder(x2))
        # NOTE here should be in no_grad mode
        self._update_momentum_params()
        target_proj1 = stop_gradient(self.momentum_encoder(x1))
        target_proj2 = stop_gradient(self.momentum_encoder(x2))

        return online_pred1, online_pred2, target_proj1, target_proj2


# if __name__=="__main__":
#     from ssvos.models.backbones import VideoTransformerNetwork
#     from mindspore import context
#     from mindspore.common.initializer import initializer, Normal

#     context.set_context(device_target='GPU', mode=context.PYNATIVE_MODE)

#     vtn = VideoTransformerNetwork()
#     byol = NetWithSymmetricLoss(net=BYOL(vtn), loss_fn=CosineSimilarityLoss())
#     dummy_input_1 = initializer(Normal(), (2,3,8,224,224))
#     dummy_input_2 = dummy_input_1.copy()
#     print(byol(dummy_input_1, dummy_input_2))
