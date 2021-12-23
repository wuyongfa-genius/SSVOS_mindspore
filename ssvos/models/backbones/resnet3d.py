"""Implement 3D ResNet backbone in MindSpore, mostly adapted from mmaction2."""
from mindspore import nn, ops
from mindspore.nn import cell
from ssvos.utils.module_utils import ConvModule
from ssvos.utils.param_utils import default_weight_init, init_param


class BasicBlock3d(nn.Cell):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 ):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return self.relu(out)


class Bottleneck3d(nn.Cell):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 inflate=True,
                 inflate_style='3x1x1',
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),):
        super().__init__()
        assert inflate_style in ['3x1x1', '3x3x3']

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.conv1_stride_s = 1
        self.conv2_stride_s = spatial_stride
        self.conv1_stride_t = 1
        self.conv2_stride_t = temporal_stride

        if self.inflate:
            if inflate_style == '3x1x1':
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class ResNet3d(nn.Cell):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stage_blocks (tuple | None): Set number of stages for each res layer.
            Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Default: ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: ``(3, 7, 7)``.
        conv1_stride_s (int): Spatial stride of the first conv layer.
            Default: 2.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_s (int): Spatial stride of the first pooling layer.
            Default: 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        with_pool2 (bool): Whether to use pool2. Default: True.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (1, 1, 1, 1).
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        conv_cfg (dict): Config for conv layers. required keys are ``type``
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``.
            Default: ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU', inplace=True)``.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
    """

    arch_settings = {
        # 18: (BasicBlock3d, (2, 2, 2, 2)),
        # 34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 stage_blocks=None,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(3, 7, 7),
                 conv1_stride_s=2,
                 conv1_stride_t=1,
                 pool1_stride_s=2,
                 pool1_stride_t=1,
                 with_pool1=True,
                 with_pool2=True,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 zero_init_residual=True,
                 **kwargs):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages

        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool1 = with_pool1
        self.with_pool2 = with_pool2
        self.stage_inflations = inflate  # 4 stages by default
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = self.base_channels

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self._make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                **kwargs)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.insert_child_to_cell(layer_name, res_layer)
            self.res_layers.append(res_layer)

        self.global_pool = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

        self._init_weights()

    def _init_weights(self):
        default_weight_init(self)
        # init last bn's weight to 0 so that the residual path is initialized to 0
        if self.zero_init_residual:
            for cell in self.cells():
                if isinstance(cell, Bottleneck3d):
                    init_param(cell.conv3.norm.gamma, 'zeros')
                elif isinstance(cell, BasicBlock3d):
                    init_param(cell.conv2.norm.gamma, 'zeros')
        # rename Parameter so that it doen't conflict (necessary when in GRAPH_MODE)
        print('[INFO] RENAME Parameters so that their names do not conflict.')
        for i in range(1, self.num_stages+1):
            layer_name = f'layer{i}'
            layer_cell:nn.Cell = getattr(self, layer_name)
            for name, cell in layer_cell.cells_and_names():
                if isinstance(cell, nn.Conv3d):
                    cell.weight.name = f'{layer_name}.{cell.weight.name}'
                    if cell.bias is not None:
                        cell.bias.name = f'{layer_name}.{cell.bias.name }'
                elif isinstance(cell, nn.BatchNorm3d):
                    if cell.bn2d.gamma is not None:
                        cell.bn2d.gamma.name = f'{layer_name}.{cell.bn2d.gamma.name}'
                    if cell.bn2d.beta is not None:
                        cell.bn2d.beta.name = f'{layer_name}.{cell.bn2d.beta.name}'
                    cell.bn2d.moving_mean.name = f'{layer_name}.{cell.bn2d.moving_mean.name}'
                    cell.bn2d.moving_variance.name = f'{layer_name}.{cell.bn2d.moving_variance.name}'

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=tuple([(k - 1) // 2 for k in self.conv1_kernel]),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.maxpool = ops.MaxPool3D(
            kernel_size=(1, 3, 3),
            strides=(self.pool1_stride_t, self.pool1_stride_s,
                     self.pool1_stride_s),
            pad_mode='pad',
            pad_list=(0, 0, 1, 1, 1, 1))

        self.pool2 = ops.MaxPool3D(kernel_size=(2, 1, 1), strides=(
            2, 1, 1), pad_mode='pad', pad_list=0)

    @staticmethod
    def _make_res_layer(block,
                        inplanes,
                        planes,
                        blocks,
                        spatial_stride=1,
                        temporal_stride=1,
                        dilation=1,
                        inflate=1,
                        inflate_style='3x1x1',
                        norm_cfg=None,
                        act_cfg=None,
                        conv_cfg=None,
                        **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x1x1'.
            conv_cfg (dict | None): Config for norm layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate,
                                            int) else (inflate, ) * blocks
        assert len(inflate) == blocks
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvModule(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                **kwargs))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    **kwargs))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        if self.with_pool1:
            x = self.maxpool(x)
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            if len(self.out_indices) == 1 and self.out_indices[0] == self.num_stages-1:
                return self.flatten(self.global_pool(outs[0], (2, 3, 4)))
            return outs[0]

        return tuple(outs)


# if __name__=="__main__":
#     from mindspore import context
#     from mindspore.common.initializer import initializer, Normal

#     context.set_context(device_target='GPU', mode=context.GRAPH_MODE)
#     resnet = ResNet3d(depth=50)
#     dummy_input = initializer(Normal(), (1,3,8,224,224))
#     y = resnet(dummy_input)
#     print(y.shape)