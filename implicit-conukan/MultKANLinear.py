import torch
import torch.nn.functional as F
import math


class MultKANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            sum_features,
            mult_features,
            mult_arity=2,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(MultKANLinear, self).__init__()
        self.in_features = in_features
        self.sum_features = sum_features
        self.mult_features = mult_features
        self.out_features = sum_features + mult_features
        self.mult_arity = mult_arity if isinstance(mult_arity, list) else [mult_arity] * mult_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(self.sum_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(self.sum_features, in_features, grid_size + spline_order)
        )

        self.base_weight_mult = torch.nn.Parameter(torch.Tensor(self.mult_features, in_features))
        self.spline_weight_mult = torch.nn.Parameter(
            torch.Tensor(self.mult_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(self.sum_features, in_features)
            )
            self.spline_scaler_mult = torch.nn.Parameter(
                torch.Tensor(self.mult_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.sum_features, self.in_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            x = self.grid[self.spline_order: -self.spline_order]
            coeff = self.curve2coeff(
                x,
                noise,
            )

            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * coeff
            )

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

        torch.nn.init.kaiming_uniform_(self.base_weight_mult, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise_mult = (
                (
                    torch.rand(self.grid_size + 1, self.mult_features, self.in_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            x = self.grid[self.spline_order: -self.spline_order]
            coeff_mult = self.curve2coeff(
                x,
                noise_mult,
            )

            self.spline_weight_mult.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * coeff_mult
            )

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler_mult, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2

        in_features = x.size(1)

        x = x.unsqueeze(-1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)

        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            denom1 = grid[:, :, k:-1] - grid[:, :, :-(k + 1)]
            denom2 = grid[:, :, k + 1:] - grid[:, :, 1:-k]

            numer1 = x - grid[:, :, :-(k + 1)]
            numer2 = grid[:, :, k + 1:] - x

            bases = (
                    (numer1 / denom1) * bases[:, :, :-1]
                    + (numer2 / denom2) * bases[:, :, 1:]
            )

        bases = bases.contiguous()
        assert bases.size() == (
            x.size(0),
            in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"x should be 1D tensor, but got x.shape = {x.shape}")

        A = self.b_splines(x).squeeze(1)
        if y.shape[0] != A.shape[0]:
            raise ValueError(
                f"The first dimension of y should match the number of rows in A, but got y.shape = {y.shape}, A.shape = {A.shape}")

        out_features = y.shape[1]
        in_features = y.shape[2]
        coeff = torch.zeros(out_features, in_features, self.grid_size + self.spline_order, device=y.device)

        for in_idx in range(in_features):
            y_in = y[:, :, in_idx]
            solution = torch.linalg.lstsq(A, y_in).solution
            coeff[:, in_idx, :] = solution.t()

        return coeff

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    @property
    def scaled_spline_weight_mult(self):
        return self.spline_weight_mult * (
            self.spline_scaler_mult.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_activation = self.base_activation(x)

        base_output = F.linear(base_activation, self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.sum_features, -1),
        )
        sum_output = base_output + spline_output  # (batch_size, sum_features)

        base_output_mult = F.linear(base_activation, self.base_weight_mult)
        spline_output_mult = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight_mult.view(self.mult_features, -1),
        )
        mult_input = base_output_mult + spline_output_mult  # (batch_size, total_mult_inputs)

        mult_outputs = []
        idx = 0
        for arity in self.mult_arity:
            mult_output = torch.prod(mult_input[:, idx: idx + arity], dim=1, keepdim=True)
            mult_outputs.append(mult_output)
            idx += arity
        mult_output = torch.cat(mult_outputs, dim=1)  # (batch_size, mult_features)

        output = torch.cat([sum_output, mult_output], dim=1)  # (batch_size, out_features)
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_sum = self.spline_weight.abs().mean(-1)
        regularization_loss_activation_sum = l1_sum.sum()
        p_sum = l1_sum / regularization_loss_activation_sum
        regularization_loss_entropy_sum = -torch.sum(p_sum * p_sum.log())

        l1_mult = self.spline_weight_mult.abs().mean(-1)
        regularization_loss_activation_mult = l1_mult.sum()
        p_mult = l1_mult / regularization_loss_activation_mult
        regularization_loss_entropy_mult = -torch.sum(p_mult * p_mult.log())

        return (
            regularize_activation * (regularization_loss_activation_sum + regularization_loss_activation_mult)
            + regularize_entropy * (regularization_loss_entropy_sum + regularization_loss_entropy_mult)
        )
