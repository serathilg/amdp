import torch
from torch import nn
from torch.distributions.normal import Normal

from src.torchrl.reppo_util import hl_gauss

# from .helpers import SinusoidalPosEmb, cosine_beta_schedule, linear_beta_schedule, vp_beta_schedule, extract, Losses
# from .diffusion_utils import Progress, Silent

def get_activation(name):
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "swish":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}")


def normed_activation_layer(
    in_features, out_features, use_norm=True, activation="swish", device=None
):
    layers = [nn.Linear(in_features, out_features, device=device)]
    if use_norm:
        layers.append(nn.RMSNorm([out_features], device=device))
        # layers.append(nn.LayerNorm([out_features], device=device)) # for torch 2.4.0
    if activation is not None:
        layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class FCNN(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_dim=256,
        hidden_activation="swish",
        output_activation=None,
        use_norm=True,
        use_output_norm=False,
        layers=2,
        input_activation=False,
        device=None,
    ):
        super().__init__()
        net = []
        if layers == 1:
            net.append(
                normed_activation_layer(
                    in_features,
                    out_features,
                    use_norm=use_output_norm,
                    activation=output_activation,
                    device=device,
                )
            )
        else:
            if input_activation:
                net.append(get_activation(hidden_activation))
            net.append(
                normed_activation_layer(
                    in_features,
                    hidden_dim,
                    use_norm=use_norm,
                    activation=hidden_activation,
                    device=device,
                )
            )
            for _ in range(layers - 2):
                net.append(
                    normed_activation_layer(
                        hidden_dim,
                        hidden_dim,
                        use_norm=use_norm,
                        activation=hidden_activation,
                        device=device,
                    )
                )
            net.append(
                normed_activation_layer(
                    hidden_dim,
                    out_features,
                    use_norm=use_output_norm,
                    activation=output_activation,
                    device=device,
                )
            )
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        hidden_dim=256,
        use_norm=True,
        use_encoder_norm=False,
        encoder_layers=1,
        head_layers=1,
        pred_layers=1,
        device=None,
    ):
        super().__init__()
        self.feature_module = FCNN(
            in_features=n_obs + n_act,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=use_encoder_norm,
            layers=encoder_layers,
            device=device,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=1,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=head_layers,
            device=device,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=pred_layers,
            device=device,
        )

    def features(self, obs, action):
        state = torch.cat([obs, action], dim=-1)
        return self.feature_module(state)

    def critic_head(self, features):
        return self.critic_module(features)

    def critic(self, obs, action):
        features = self.features(obs, action)
        return self.critic_head(features)

    def forward(self, obs, action):
        features = self.features(obs, action)
        return self.pred_module(features)


class Critic(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        num_atoms: int,
        vmin: float,
        vmax: float,
        hidden_dim=256,
        use_norm=True,
        use_encoder_norm=False,
        encoder_layers=1,
        head_layers=1,
        pred_layers=1,
        device=None,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.hidden_dim = hidden_dim
        self.feature_module = FCNN(
            in_features=n_obs + n_act,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=use_encoder_norm,
            layers=encoder_layers,
            device=device,
        )
        self.critic_module = FCNN(
            in_features=hidden_dim,
            out_features=num_atoms,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            input_activation=True,
            layers=head_layers,
            device=device,
        )
        self.pred_module = FCNN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            input_activation=True,
            use_output_norm=False,
            layers=pred_layers,
            device=device,
        )
        self.values = torch.linspace(
            vmin, vmax, num_atoms, device=device, dtype=torch.float32
        )
        zeros = hl_gauss(
            torch.zeros(1, device=device), self.vmin, self.vmax, self.num_atoms
        )
        zeros.requires_grad = True
        self.zero_dist = nn.Parameter(
            hl_gauss(
                torch.zeros(1, device=device), self.vmin, self.vmax, self.num_atoms
            )
        )

    def forward(self, obs, action):
        inp = torch.cat([obs, action], dim=-1)
        features = self.feature_module(inp)
        next_pred = self.pred_module(features)
        logits = self.critic_module(features) + 40.9 * self.zero_dist
        value_cats = torch.softmax(logits, dim=-1)
        value = value_cats @ self.values
        return value, logits, next_pred, features


class Actor(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        ent_start: float,
        kl_start: float,
        hidden_dim=256,
        use_norm=True,
        layers=2,
        min_std=0.1,
        device=None,
    ):
        super().__init__()
        self.model = FCNN(
            in_features=n_obs,
            out_features=2 * n_act,
            hidden_dim=hidden_dim,
            hidden_activation="swish",
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            layers=layers,
            device=device,
        )
        self.log_temp = nn.Parameter(
            torch.log(torch.tensor(ent_start, device=device, dtype=torch.float32))
        )
        self.log_lagrange = nn.Parameter(
            torch.log(torch.tensor(kl_start, device=device, dtype=torch.float32))
        )
        self.min_std = min_std

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        x = self.model(obs)
        mean, log_std = torch.split(x, x.shape[-1] // 2, dim=-1)
        std = torch.exp(log_std) + self.min_std
        pi = Normal(mean, std, validate_args=False)

        transformed_pi = torch.distributions.TransformedDistribution(
            pi, [torch.distributions.TanhTransform()]
        )
        return (
            transformed_pi,
            torch.tanh(mean),
            torch.exp(self.log_temp),
            torch.exp(self.log_lagrange),
        )
