
import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor, center: bool = True) -> torch.Tensor:
        if x.shape[-1:] != self._mean.shape[-1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[-1:]}), got {x.shape}"
            )

        if self.training:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        x = x.flatten(end_dim=-2)

        if self.until is not None and self.count >= self.until:
            return

        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)

        # Update count
        new_count = self.count + batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean += (batch_size / new_count) * delta

        # Update variance using Chan's parallel algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        if self.count > 0:  # Ensure we're not dividing by zero
            batch_var = torch.mean((x - batch_mean) ** 2, dim=0, keepdim=True)
            delta2 = batch_mean - self._mean
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta2**2) * (self.count * batch_size / new_count)
            self._var = M2 / new_count
        else:
            # For first batch, just use batch variance
            self._var = torch.mean((x - self._mean) ** 2, dim=0, keepdim=True)

        self._std = torch.sqrt(self._var)
        self.count = new_count

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


def hl_gauss(inp, vmin, vmax, num_atoms):
    x = torch.clip(inp, vmin, max=vmax)
    bin_width = (vmax - vmin) / (num_atoms - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = torch.linspace(
        vmin - bin_width / 2,
        vmax + bin_width / 2,
        num_atoms + 1,
        device=inp.device,
    )
    sigma = bin_width * sigma_to_final_sigma_ratio
    cdf_evals = torch.erf(
        (support.unsqueeze(0) - x).squeeze()
        / (torch.sqrt(torch.tensor(2.0)) * sigma + 1e-6)
    )
    z = cdf_evals[..., -1] - cdf_evals[..., 0]
    target_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    target_probs = (target_probs / (z.unsqueeze(-1) + 1e-6)).reshape(
        *inp.shape[:-1], num_atoms
    )

    return target_probs
