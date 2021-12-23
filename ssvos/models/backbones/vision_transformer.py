"""Implement Vision Transformer in MindSpore."""
from mindspore import nn, ops
from mindspore.common.initializer import TruncatedNormal
from ssvos.utils.param_utils import init_param


class MLP(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        keep_drop_probs = (1-drop, 1-drop)

        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(keep_drop_probs[0])
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop2 = nn.Dropout(keep_drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim//num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Dense(dim, dim*3, has_bias=qkv_bias)
        
        self.softmax = ops.Softmax(axis=-1)
        self.attn_drop = nn.Dropout(keep_prob=1-attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1-proj_drop)
        # init some opreations to be used
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.chunk = ops.Split(axis=0, output_num=3)  # chunk qkv into q,k,v
        self.bmm_qk = ops.BatchMatMul(transpose_b=True) # q*k^T
        self.bmm_av = ops.BatchMatMul() # attention*v

    def construct(self, x):
        B, N, C = x.shape
        # project x to q, k, v
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv
        # compute attention scores
        attention = self.bmm_qk(q, k) * self.scale
        attention = self.softmax(attention)
        attention = self.attn_drop(attention)
        # get new x by attend v using attention scores
        x = self.bmm_av(attention, v)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (B, N, C))
        # project back to x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        norm_dim = dim if isinstance(dim, (tuple, list)) else (dim,)
        self.norm1 = norm_layer(norm_dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(norm_dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def construct(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Cell):
    def __init__(self, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4.,qkv_bias=True,
                drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.encoder_layers = nn.SequentialCell([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])
        self._init_weights()
    
    def _init_weights(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                init_param(cell.weight, TruncatedNormal, sigma=.02)
                if cell.bias is not None:
                    init_param(cell.bias, 'zeros')
            elif isinstance(cell, nn.LayerNorm):
                init_param(cell.gamma, 'ones')
                init_param(cell.beta, 'zeros')

    def construct(self, x):
        return self.encoder_layers(x)


# if __name__=="__main__":
#     from mindspore import context
#     from mindspore.common.initializer import initializer, Normal

#     context.set_context(device_target='GPU', mode=context.GRAPH_MODE)
#     transformer = TransformerEncoder()
#     dummy_input = initializer(Normal(), (2, 8, 768))
#     y = transformer(dummy_input)
#     print(y.shape)