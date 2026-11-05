import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from typing import Literal, Tuple


class FeatureProjector(nnx.Module):
    """
    Helper module to process raw inputs (time or actions) into embeddings.
    Supports:
    - Linear vs MLP mode.
    - Fourier features with configurable frequency ranges.
    - Mixing linear and fourier features as MLP inputs.
    """

    def __init__(
        self,
        input_dim: int,
        mode: Literal["linear", "mlp"] = "mlp",
        mlp_input_type: Literal["linear", "fourier", "both"] = "both",
        num_fourier_feats: int = 32,
        fourier_range: Tuple[float, float] = (0.1, 100.0),
        num_hid: int = 64,
        num_out: int = 16,
        *,
        rngs: nnx.Rngs,
    ):
        self.mode = mode
        self.mlp_input_type = mlp_input_type
        self.input_dim = input_dim

        if mode == "linear":
            self.output_dim = input_dim
            return

        self.num_fourier_feats = num_fourier_feats
        freqs = jnp.geomspace(fourier_range[0], fourier_range[1], num_fourier_feats)
        # singleton dim for broadcasting with input_dim
        self.coeff = nnx.Variable(freqs[None, :])
        # Leave phase trainable per input_dim
        self.phase = nnx.Param(jnp.zeros((input_dim, num_fourier_feats)))

        # sin and cos per frequency per input dim
        fourier_dim = input_dim * num_fourier_feats * 2

        if mlp_input_type == "linear":
            mlp_in_dim = input_dim
        elif mlp_input_type == "fourier":
            mlp_in_dim = fourier_dim
        else:  # "both"
            mlp_in_dim = fourier_dim + input_dim

        self.mlp = nnx.Sequential(
            nnx.Linear(mlp_in_dim, num_hid, rngs=rngs),
            nnx.gelu,
            nnx.Linear(num_hid, num_out, rngs=rngs),
        )
        self.output_dim = num_out

    def _get_fourier(self, x):
        # x shape: (input_dim,)
        # Reshape for broadcasting with coeff (input_dim, freqs)
        x_expanded = x[:, None]
        angle = (x_expanded * self.coeff.value) + self.phase.value
        # (input_dim, num_fourier_feats * 2)
        feats = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        # Flatten to (input_dim * num_fourier_feats * 2)
        return feats.reshape(-1)

    def __call__(self, x):
        # x may be scalar instead of 1D
        x = jnp.atleast_1d(x)

        if self.mode == "linear":
            return x

        # Prepare MLP input
        if self.mlp_input_type == "linear":
            mlp_in = x
        elif self.mlp_input_type == "fourier":
            mlp_in = self._get_fourier(x)
        else:  # both
            mlp_in = jnp.concatenate([x, self._get_fourier(x)], axis=-1)

        return self.mlp(mlp_in)


class ControlNetwork(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        num_layers: int = 2,
        num_hid: int = 64,
        # Time configuration
        time_mode: Literal["linear", "mlp"] = "mlp",
        time_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_time_fourier: int = 16,
        time_fourier_range_min: float = 0.1,
        time_fourier_range_max: float = 100.0,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        # Action configuration
        action_mode: Literal["linear", "mlp"] = "linear",
        action_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_action_fourier: int = 16,
        action_fourier_range_min: float = 0.1,
        action_fourier_range_max: float = 100.0,
        num_action_hid: int = 32,
        num_action_out: int = 16,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        layer_norm: bool = False,
        layer_norm_type: str = "LayerNorm",
        *,
        rngs: nnx.Rngs,
    ):
        self.outer_clip = outer_clip

        time_fourier_range = (time_fourier_range_min, time_fourier_range_max)
        # Time encoder network
        self.time_projector = FeatureProjector(
            input_dim=1,
            mode=time_mode,
            mlp_input_type=time_mlp_input,
            num_fourier_feats=num_time_fourier,
            fourier_range=time_fourier_range,
            num_hid=num_time_hid,
            num_out=num_time_out,
            rngs=rngs,
        )

        action_fourier_range = (action_fourier_range_min, action_fourier_range_max)
        # Action encoder network
        self.action_projector = FeatureProjector(
            input_dim=action_dim,
            mode=action_mode,
            mlp_input_type=action_mlp_input,
            num_fourier_feats=num_action_fourier,
            fourier_range=action_fourier_range,
            num_hid=num_action_hid,
            num_out=num_action_out,
            rngs=rngs,
        )

        # State-time network
        # Dynamically calculate input size
        combined_input_dim = self.action_projector.output_dim + observation_dim + self.time_projector.output_dim

        layers = []
        curr_dim = combined_input_dim

        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_dim = action_dim if is_last else num_hid

            layer = nnx.Linear(curr_dim, out_dim, rngs=rngs)

            if is_last:
                # custom init but no norm or activation afterwards
                layer.kernel.value *= weight_init
                layer.bias.value = jnp.full_like(layer.bias.value, bias_init)
                layers.append(layer)
            else:
                layers.append(layer)
                if layer_norm:
                    layers.append(getattr(nnx, layer_norm_type)(out_dim, rngs=rngs))
                layers.append(nnx.gelu)

            curr_dim = out_dim

        self.state_time_net = nnx.Sequential(*layers)

    def __call__(
        self,
        actions: Float[Array, "act"],
        observations: Float[Array, "obs"],
        time: Float[Array, ""],
    ) -> Float[Array, "act"]:
        """Per-sample control u(x, obs, t); time is a scalar in [0, 1]."""
        t_scaled = (time * 2.0) - 1.0

        t_emb = self.time_projector(t_scaled)
        a_emb = self.action_projector(actions)

        x = jnp.concatenate([a_emb, observations, t_emb], axis=-1)
        out = self.state_time_net(x)

        return jnp.clip(out, -self.outer_clip, self.outer_clip)

class ResidualBlock(nnx.Module):
    """
    A modern residual block supporting AdaLN conditioning, Gated GELU, 
    and SimBAv2-style magnitude shift preservation.
    """
    def __init__(
        self,
        stream_dim: int,
        context_dim: int | None = None,
        cond_mode: Literal["concat", "adaln"] = "adaln",
        activation: Literal["gelu", "gated_gelu"] = "gated_gelu",
        layer_norm_type: str = "RMSNorm",
        simba_shift: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.cond_mode = cond_mode
        self.activation = activation
        self.simba_shift = simba_shift

        # Pre-Normalization layer
        self.norm = getattr(nnx, layer_norm_type)(stream_dim, rngs=rngs)

        # AdaLN Modulation Projections
        if cond_mode == "adaln":
            assert context_dim is not None, "context_dim must be provided for adaln mode."
            self.adaln_proj = nnx.Linear(
                context_dim, 
                2 * stream_dim, 
                kernel_init=jax.nn.initializers.zeros, # Initialize to 0 for stable start
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs
            )

        # Simbav2 prepends/appends a constant 1 to the features to retain magnitude shift
        mlp_in_dim = stream_dim + 1 if simba_shift else stream_dim

        # MLP Core
        if activation == "gelu":
            hidden_dim = int(4 * stream_dim) # Standard 4d expansion
            self.up_proj = nnx.Linear(mlp_in_dim, hidden_dim, rngs=rngs)
        elif activation == "gated_gelu":
            hidden_dim = int((8 / 3) * stream_dim) # Parameter-matched expansion
            self.gate_proj = nnx.Linear(mlp_in_dim, hidden_dim, rngs=rngs)
            self.up_proj = nnx.Linear(mlp_in_dim, hidden_dim, rngs=rngs)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Final projection: initialized to zero so the block is exactly the identity function at init
        self.down_proj = nnx.Linear(
            hidden_dim, 
            stream_dim, 
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs
        )

    def __call__(self, x, context=None):
        # 1. Pre-Norm
        h = self.norm(x)

        # 2. Conditioning
        if self.cond_mode == "adaln":
            scale_shift = self.adaln_proj(context)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            h = h * (1.0 + scale) + shift

        # 3. SimBAv2 magnitude preserving shift (appends a 1 to the feature vector)
        if self.simba_shift:
            # Handles unbatched (dim,) and batched (B, dim) automatically
            ones = jnp.ones((*h.shape[:-1], 1), dtype=h.dtype)
            h = jnp.concatenate([h, ones], axis=-1)

        # 4. Activation Expansion
        if self.activation == "gelu":
            h = self.up_proj(h)
            h = nnx.gelu(h)
        elif self.activation == "gated_gelu":
            gate = nnx.gelu(self.gate_proj(h))
            up = self.up_proj(h)
            h = gate * up

        # 5. Down-projection & Residual connection
        h = self.down_proj(h)
        return x + h


class ResidualControlNetwork(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        # Residual Architecture config
        stream_dim: int = 128,
        num_blocks: int = 4,
        cond_mode: Literal["concat", "adaln"] = "adaln",
        context_dim: int = 128,
        activation: Literal["gelu", "gated_gelu"] = "gated_gelu",
        layer_norm_type: str = "RMSNorm",  # residual-stream convention (SimBaV2-style); the trainers pass the config value explicitly
        simba_shift: bool = False,
        # Time configuration
        time_mode: Literal["linear", "mlp"] = "mlp",
        time_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_time_fourier: int = 16,
        time_fourier_range_min: float = 0.1,
        time_fourier_range_max: float = 100.0,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        # Action configuration
        action_mode: Literal["linear", "mlp"] = "linear",
        action_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_action_fourier: int = 16,
        action_fourier_range_min: float = 0.1,
        action_fourier_range_max: float = 100.0,
        num_action_hid: int = 32,
        num_action_out: int = 16,
        # Output config
        outer_clip: float = 1e4,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.outer_clip = outer_clip
        self.cond_mode = cond_mode

        # Time encoder network
        self.time_projector = FeatureProjector(
            input_dim=1,
            mode=time_mode,
            mlp_input_type=time_mlp_input,
            num_fourier_feats=num_time_fourier,
            fourier_range=(time_fourier_range_min, time_fourier_range_max),
            num_hid=num_time_hid,
            num_out=num_time_out,
            rngs=rngs,
        )

        # Action encoder network
        self.action_projector = FeatureProjector(
            input_dim=action_dim,
            mode=action_mode,
            mlp_input_type=action_mlp_input,
            num_fourier_feats=num_action_fourier,
            fourier_range=(action_fourier_range_min, action_fourier_range_max),
            num_hid=num_action_hid,
            num_out=num_action_out,
            rngs=rngs,
        )

        # Determine architecture topology based on Conditioning Mode
        if cond_mode == "concat":
            combined_input_dim = self.action_projector.output_dim + observation_dim + self.time_projector.output_dim
            self.initial_proj = nnx.Linear(combined_input_dim, stream_dim, rngs=rngs)
            
        elif cond_mode == "adaln":
            # For AdaLN, we process the action/state into the main residual stream.
            self.initial_proj = nnx.Linear(self.action_projector.output_dim, stream_dim, rngs=rngs)
            
            # Shared MLP processing observations and time to build a contextual representation
            context_input_dim = observation_dim + self.time_projector.output_dim
            self.context_mlp = nnx.Sequential(
                nnx.Linear(context_input_dim, context_dim, rngs=rngs),
                nnx.gelu,
                nnx.Linear(context_dim, context_dim, rngs=rngs)
            )
        else:
            raise ValueError(f"Unknown condition mode: {cond_mode}")

        # Construct Residual Blocks
        self.blocks =[
            ResidualBlock(
                stream_dim=stream_dim,
                context_dim=context_dim if cond_mode == "adaln" else None,
                cond_mode=cond_mode,
                activation=activation,
                layer_norm_type=layer_norm_type,
                simba_shift=simba_shift,
                rngs=rngs
            )
            for _ in range(num_blocks)
        ]

        # Final projection sequence
        self.final_norm = getattr(nnx, layer_norm_type)(stream_dim, rngs=rngs)
        self.final_proj = nnx.Linear(stream_dim, action_dim, rngs=rngs)
        
        # Apply custom initialization to the last layer so the initial network predictions are close to zero
        self.final_proj.kernel.value *= weight_init
        self.final_proj.bias.value = jnp.full_like(self.final_proj.bias.value, bias_init)

    def __call__(
        self,
        actions: Float[Array, "act"],
        observations: Float[Array, "obs"],
        time: Float[Array, ""],
    ) -> Float[Array, "act"]:
        """Per-sample control u(x, obs, t); time is a scalar in [0, 1]."""
        t_scaled = (time * 2.0) - 1.0

        t_emb = self.time_projector(t_scaled)
        a_emb = self.action_projector(actions)

        # 1. Start the feature stream
        if self.cond_mode == "concat":
            x = jnp.concatenate([a_emb, observations, t_emb], axis=-1)
            h = self.initial_proj(x)
            context = None
        else:  # adaln
            h = self.initial_proj(a_emb)
            c_in = jnp.concatenate([observations, t_emb], axis=-1)
            context = self.context_mlp(c_in)

        # 2. Pass through residual blocks
        for block in self.blocks:
            h = block(h, context)

        # 3. Final norm and projection
        h = self.final_norm(h)
        out = self.final_proj(h)

        return jnp.clip(out, -self.outer_clip, self.outer_clip)