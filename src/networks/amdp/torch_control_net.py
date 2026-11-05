import torch
from torch import nn

class ControlNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        num_layers: int = 2,
        num_hid: int = 64,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        layer_norm: bool = False,
        layer_norm_type: str = "LayerNorm",
        device=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.layer_norm = layer_norm
        self.layer_norm_type = layer_norm_type
        self.num_layers = num_layers
        self.num_hid = num_hid
        self.num_time_hid = num_time_hid
        self.num_time_out = num_time_out
        self.outer_clip = outer_clip
        self.inner_clip = inner_clip
        self.weight_init = weight_init
        self.bias_init = bias_init

        # Initialize timestep parameters
        self.timestep_phase = nn.Parameter(torch.zeros(1, self.num_time_hid, device=device))
        # Store timestep_coeff as a buffer (non-trainable parameter)
        self.register_buffer(
            'timestep_coeff', 
            torch.linspace(0.1, 100, self.num_time_hid, device=device).unsqueeze(0)
        )

        # Time encoder network
        self.time_coder_state = nn.Sequential(
            nn.Linear(self.num_time_hid * 2, self.num_time_hid, device=device),
            nn.GELU(),
            nn.Linear(self.num_time_hid, self.num_time_out, device=device),
        )

        # State-time network
        layers = []
        layers.extend(
            [
                nn.Linear(
                    self.action_dim + self.observation_dim + self.num_time_out,
                    self.num_hid,
                    device=device,
                ),
                nn.GELU(),
            ]
        )

        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.num_hid, self.num_hid, device=device))
            if self.layer_norm:
                if self.layer_norm_type == "LayerNorm":
                    layers.append(nn.LayerNorm(self.num_hid, device=device))
                elif self.layer_norm_type == "RMSNorm":
                    layers.append(nn.RMSNorm(self.num_hid, device=device))
            layers.append(nn.GELU())

        # Output layer with custom initialization
        output_layer = nn.Linear(self.num_hid, self.action_dim, device=device)
        # Apply custom initialization
        with torch.no_grad():
            output_layer.weight.data *= self.weight_init
            output_layer.bias.data.fill_(self.bias_init)
        layers.append(output_layer)

        self.state_time_net = nn.Sequential(*layers)

    def get_fourier_features(self, timesteps):
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return torch.cat([sin_embed_cond, cos_embed_cond], dim=-1)

    def forward(self, actions, observations, time):
        time_emb = self.get_fourier_features(time)
        t_net = self.time_coder_state(time_emb)

        # repeat to match actions
        t_net = t_net.expand(actions.size(0), -1)

        extended_input = torch.cat((actions, observations, t_net), dim=-1)
        out_state = self.state_time_net(extended_input)
        out_state = torch.clamp(out_state, -self.outer_clip, self.outer_clip)
        return out_state