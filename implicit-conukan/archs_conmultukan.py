import math
import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_

from .MultKANLinear import MultKANLinear
from .building_blocks2D import (
    ConvSODEFunc,
    InitialVelocity,
    ODEBlock,
    get_nonlinearity,
)

__file__ = ["conMultUKAN"]

MAX_NUM_STEPS = 1000


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultKANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False,
                 grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1],
                 enable_standalone_scale_spline=True,
                 sum_features_fc1=None, sum_features_fc2=None, sum_features_fc3=None,
                 mult_features_fc1=None, mult_features_fc2=None, mult_features_fc3=None,
                 mult_arity_fc1=2, mult_arity_fc2=2, mult_arity_fc3=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        if not no_kan:
            # Default to all sum nodes if mult_features not specified
            if sum_features_fc1 is None and mult_features_fc1 is None:
                sum_features_fc1 = hidden_features // 4
                mult_features_fc1 = hidden_features - sum_features_fc1
            if sum_features_fc2 is None and mult_features_fc2 is None:
                sum_features_fc2 = out_features // 4
                mult_features_fc2 = out_features - sum_features_fc2
            if sum_features_fc3 is None and mult_features_fc3 is None:
                sum_features_fc3 = out_features // 4
                mult_features_fc3 = out_features - sum_features_fc3

            self.fc1 = MultKANLinear(
                in_features=in_features,
                sum_features=sum_features_fc1,
                mult_features=mult_features_fc1,
                mult_arity=mult_arity_fc1,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
            )
            fc1_output_dim = sum_features_fc1 + mult_features_fc1

            self.fc2 = MultKANLinear(
                in_features=fc1_output_dim,
                sum_features=sum_features_fc2,
                mult_features=mult_features_fc2,
                mult_arity=mult_arity_fc2,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
            )
            fc2_output_dim = sum_features_fc2 + mult_features_fc2

            self.fc3 = MultKANLinear(
                in_features=fc1_output_dim,
                sum_features=sum_features_fc3,
                mult_features=mult_features_fc3,
                mult_arity=mult_arity_fc3,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
                enable_standalone_scale_spline=enable_standalone_scale_spline,
            )
            fc3_output_dim = sum_features_fc3 + mult_features_fc3
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)
            fc1_output_dim = hidden_features
            fc2_output_dim = out_features
            fc3_output_dim = out_features

        self.dwconv_1 = DW_bn_relu(fc1_output_dim)
        self.dwconv_2 = DW_bn_relu(fc2_output_dim)
        self.dwconv_3 = DW_bn_relu(fc3_output_dim)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # First MultKANLinear layer
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        x = self.dwconv_1(x, H, W)

        # Save intermediate output for fc3
        x_fc3_input = x.clone()

        # Second MultKANLinear layer
        x = self.fc2(x.reshape(B * N, -1))
        x = x.reshape(B, N, -1).contiguous()
        x = self.dwconv_2(x, H, W)

        # Third MultKANLinear layer (uses the output from fc1 and dwconv_1)
        x_fc3 = self.fc3(x_fc3_input.reshape(B * N, -1))
        x_fc3 = x_fc3.reshape(B, N, -1).contiguous()
        x_fc3 = self.dwconv_3(x_fc3, H, W)

        # Combine outputs from fc2 and fc3
        x = x + x_fc3

        return x


class MultKANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False, **kwargs):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = MultKANLayer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            no_kan=no_kan,
            **kwargs
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class conMultUKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256], no_kan=False, output_dim=[32, 64, 128],
                 drop_rate=0., time_dependent=False, non_linearity="softplus",
                 tol=1e-3, adjoint=False, method="rk4", drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1],
                 **kwargs):
        super().__init__()

        nf = input_channels
        self.method = method
        print(f"Solver: {method}")

        self.initial_velocity = InitialVelocity(nf, non_linearity)

        ode_down1 = ConvSODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_down1 = ODEBlock(ode_down1, tol=tol, adjoint=adjoint)
        self.conv_down1_2 = nn.Conv2d(nf * 2, output_dim[0], 1, 1)

        ode_down2 = ConvSODEFunc(output_dim[0], time_dependent, non_linearity)
        self.odeblock_down2 = ODEBlock(ode_down2, tol=tol, adjoint=adjoint)
        self.conv_down2_3 = nn.Conv2d(output_dim[0], output_dim[1], 1, 1)

        ode_down3 = ConvSODEFunc(output_dim[1], time_dependent, non_linearity)
        self.odeblock_down3 = ODEBlock(ode_down3, tol=tol, adjoint=adjoint)
        self.conv_down3_4 = nn.Conv2d(output_dim[1], output_dim[2], 1, 1)

        ode_down4 = ConvSODEFunc(output_dim[2], time_dependent, non_linearity)
        self.odeblock_down4 = ODEBlock(ode_down4, tol=tol, adjoint=adjoint)

        ode_down5 = ConvSODEFunc(embed_dims[1], time_dependent, non_linearity)
        self.odeblock_down5 = ODEBlock(ode_down5, tol=tol, adjoint=adjoint)

        ode_up1 = ConvSODEFunc(output_dim[2], time_dependent, non_linearity)
        self.odeblock_up1 = ODEBlock(ode_up1, tol=tol, adjoint=adjoint)
        self.conv_up1_2 = nn.Conv2d(output_dim[2] * 2, output_dim[1], 1, 1)

        ode_up2 = ConvSODEFunc(output_dim[1], time_dependent, non_linearity)
        self.odeblock_up2 = ODEBlock(ode_up2, tol=tol, adjoint=adjoint)
        self.conv_up2_3 = nn.Conv2d(output_dim[1] * 2, output_dim[0], 1, 1)

        ode_up3 = ConvSODEFunc(output_dim[0], time_dependent, non_linearity)
        self.odeblock_up3 = ODEBlock(ode_up3, tol=tol, adjoint=adjoint)
        self.conv_up3_4 = nn.Conv2d(output_dim[0] * 2, 6, 1, 1)

        ode_up4 = ConvSODEFunc(6, time_dependent, non_linearity)
        self.odeblock_up4 = ODEBlock(ode_up4, tol=tol, adjoint=adjoint)
        self.conv_up4_5 = nn.Conv2d(12, nf * 2, 1, 1)

        ode_up5 = ConvSODEFunc(nf * 2, time_dependent, non_linearity)
        self.odeblock_up5 = ODEBlock(ode_up5, tol=tol, adjoint=adjoint)

        self.classifier = nn.Conv2d(nf * 2, num_classes, 1)

        self.non_linearity = get_nonlinearity(non_linearity)

        self.norm2 = norm_layer(embed_dims[0])
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[2])
        self.dnorm4 = norm_layer(embed_dims[1])
        self.dnorm5 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.conv_bottleneck = nn.Conv2d(embed_dims[1], embed_dims[0], 1, 1)
        odetest_up = ConvSODEFunc(embed_dims[0], time_dependent, non_linearity)
        self.odeblock_bottleneck = ODEBlock(odetest_up, tol=tol, adjoint=adjoint)

        self.block0 = nn.ModuleList([MultKANBlock(
            dim=embed_dims[0],
            drop=drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **kwargs
        )])

        self.block1 = nn.ModuleList([MultKANBlock(
            dim=embed_dims[1],
            drop=drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **kwargs
        )])

        self.bottleneck_block = nn.ModuleList([MultKANBlock(
            dim=embed_dims[2],
            drop=drop_rate,
            drop_path=dpr[1],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **kwargs
        )])

        self.dblock1 = nn.ModuleList([MultKANBlock(
            dim=embed_dims[2],
            drop=drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **kwargs
        )])

        self.dblock0 = nn.ModuleList([MultKANBlock(
            dim=embed_dims[0],
            drop=drop_rate,
            drop_path=dpr[0],
            norm_layer=norm_layer,
            no_kan=no_kan,
            **kwargs
        )])

        self.patch_embed2 = PatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=64,
                                       embed_dim=128)

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=1, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        self.bottleneck = ConvLayer(embed_dims[2], embed_dims[1])

    def forward(self, x):
        B = x.shape[0]  # [1, 3, 256, 256]
        x = self.initial_velocity(x)  # [1, 6, 256, 256]

        features1 = self.odeblock_down1(x, method=self.method)  # [1, 6, 256, 256]
        x = self.non_linearity(self.conv_down1_2(features1))  # [1, 32, 256, 256]

        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )  # [1, 32, 128, 128]

        features2 = self.odeblock_down2(x, method=self.method)  # [1, 32, 128, 128]
        x = self.non_linearity(self.conv_down2_3(features2))  # [1, 64, 128, 128]

        x = nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )  # [1, 64, 64, 64]

        features3 = self.odeblock_down3(x, method=self.method)  # [1, 64, 64, 64]
        out = self.non_linearity(features3)
        out, H, W = self.patch_embed2(out)                      # [1, 1024, 128]
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        features4 = self.odeblock_down4(out, method=self.method)        # [1, 128, 32, 32]
        out = self.non_linearity(features4)
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [1, 160, 16, 16]

        out = self.odeblock_down5(out, method=self.method)
        out = self.non_linearity(out)
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.bottleneck_block):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [1, 256, 16, 16]

        out = self.bottleneck(out)  # [1, 160, 16, 16]

        out = self.non_linearity(self.conv_bottleneck(out))
        out = self.odeblock_bottleneck(out, method=self.method)
        out = nn.functional.interpolate(out, scale_factor=2, mode="bilinear")  # [1, 128, 32, 32]

        out = torch.cat((out, features4), dim=1)  # [1, 256, 32, 32]
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)  # [1, 1024, 256]
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [1, 256, 32, 32]

        out = self.non_linearity(self.conv_up1_2(out))
        out = self.odeblock_up2(out, method=self.method)  # [1, 64, 32, 32]

        out = nn.functional.interpolate(
            out, scale_factor=2, mode="bilinear", align_corners=False
        )  # [1, 64, 64, 64]

        out = torch.cat((out, features3), dim=1)  # [1, 128, 64, 64]
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)  # [1, 4096, 128]
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)

        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [1, 128, 64, 64]

        x = self.non_linearity(self.conv_up2_3(out))
        x = self.odeblock_up3(x, method=self.method)  # [1, 32, 64, 64]

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # [1, 32, 128, 128]

        x = torch.cat((x, features2), dim=1)    # [1, 64, 128, 128]
        x = self.non_linearity(self.conv_up3_4(x))
        x = self.odeblock_up4(x, method=self.method)  # [1, 6, 128, 128]

        x = nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )  # [1, 6, 256, 256]

        x = torch.cat((x, features1), dim=1)    # [1, 12, 256, 256]
        x = self.non_linearity(self.conv_up4_5(x))
        x = self.odeblock_up5(x, method=self.method)  # [1, 2, 256, 256]

        pred = self.classifier(x)  # [1, 1, 256, 256]

        return pred


def main():
    num_classes = 1
    input_channels = 3
    img_size = 256

    model = conMultUKAN(num_classes=num_classes, input_channels=input_channels, img_size=img_size,
                        embed_dims=[128, 160, 256], output_dim=[32, 64, 128])

    x = torch.randn(1, input_channels, img_size, img_size)

    with torch.no_grad():
        output = model(x)

    print("Output shape:", output.shape)
    print("Output:", output)


if __name__ == '__main__':
    main()
