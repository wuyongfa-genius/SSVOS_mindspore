"""Implement video transformer network in MindSpore"""
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal, TruncatedNormal
from ssvos.utils.module_utils import Identity
from ssvos.utils.param_utils import init_param
from ssvos.models.backbones.resnet import ResNet
from ssvos.models.backbones.vision_transformer import TransformerEncoder


class VideoTransformerNetwork(nn.Cell):
    def __init__(self, seqlength=8, d_model=2048,
                 num_layers=3, num_heads=12, mlp_ratio=2.,
                 dropout_embed=0., drop_mlp=0., drop_attn=0.,
                 frame_feat_extractor=ResNet, **extractor_kwargs):
        super().__init__()
        self.frame_feat_extractor = frame_feat_extractor(**extractor_kwargs)
        if hasattr(self.frame_feat_extractor, 'fc'):
            setattr(self.frame_feat_extractor, 'fc', Identity())
        frame_feat_dim = self.frame_feat_extractor.feat_dim
        assert frame_feat_dim==d_model
        self.feat_dim = d_model
        # learnable pos embed
        self.pos_emb = mindspore.Parameter(
            initializer(Normal(), (1, seqlength+1, d_model)))
        self.global_token = mindspore.Parameter(
            initializer(Normal(), (1, 1, d_model)))
        self.embed_drop = nn.Dropout(1-dropout_embed)
        self.transformer = TransformerEncoder(
            d_model, num_layers, num_heads, mlp_ratio, drop_rate=drop_mlp, attn_drop_rate=drop_attn)
        # init operations to be used
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.cat = ops.Concat(axis=1) # cat along seq axis

    def _init_weights(self):
        init_param(self.pos_emb, TruncatedNormal, sigma=.02)
        init_param(self.global_token, TruncatedNormal, sigma=.02)

    def construct(self, x):
        B, C, T, H, W = x.shape
        x = self.transpose(x, (0, 2, 1, 3, 4))
        x = self.reshape(x, B*T, C, H, W)
        frame_embeds = self.frame_feat_extractor(x)
        frame_embeds = self.reshape(frame_embeds, (B, T, frame_embeds.shape[-1]))
        # repeat global_token batch times
        global_tokens = self.global_token.repeat(B, axis=0)
        # concat global_tokens with frame_embeds
        x = self.cat(global_tokens, frame_embeds)
        # add pos embed
        x += self.pos_emb
        x = self.embed_drop(x)
        x = self.transformer(x)
        # only take global token
        global_tokens = x[:, 0]

        return global_tokens
