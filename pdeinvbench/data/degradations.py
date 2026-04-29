import torch
import typeguard
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=typeguard.typechecked)
def add_salt_and_pepper_noise(
    x: Float[torch.Tensor, "time channel xspace yspace"],
    pepper_prob: float = 0.1,
    salt_prob: float = 0.1,
    pepper_val: float = -0.5,
    salt_val: float = 0.5,
) -> Float[torch.Tensor, "time channel xspace yspace"]:
    """
    Add salt and pepper noise to the data.
    """
    no_noise_prob = 1 - pepper_prob - salt_prob
    u = torch.rand(x.shape)
    noise_arr = torch.zeros(x.shape)
    mask_pepper = (u >= no_noise_prob) & (u < no_noise_prob + pepper_prob)
    mask_salt = (u >= no_noise_prob + pepper_prob)  # rest gets salt
    noise_arr[mask_pepper] = pepper_val
    noise_arr[mask_salt] = salt_val
    return x + noise_arr


@jaxtyped(typechecker=typeguard.typechecked)
def drop_high_freq_modes(
    x: Float[torch.Tensor, "time channel xspace yspace"],
    drop_ratio: float = 0.1,
    order: int = 6,
) -> Float[torch.Tensor, "time channel xspace yspace"]:
    """
    Drop high frequency modes from the data using a Butterworth low-pass filter.
    Args:
        x: (time channel xspace yspace)
        drop_ratio: ratio of high frequency modes to drop
        order: Butterworth filter order (higher = sharper rolloff)
    Returns:
        (time channel xspace yspace)
    """
    assert 0 <= drop_ratio <= 1, "drop_ratio must be between 0 and 1"
    if drop_ratio == 0:
        return x

    X, Y = x.shape[-2:]
    device = x.device
    dtype = x.dtype
    keep_ratio = 1 - drop_ratio

    f = torch.fft.fft2(x, dim=(-2, -1))
    f = torch.fft.fftshift(f, dim=(-2, -1))

    fx = torch.fft.fftshift(torch.fft.fftfreq(X, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(Y, device=device))
    rx, ry = torch.meshgrid(fx, fy, indexing='ij')
    r = torch.sqrt(rx**2 + ry**2)

    cutoff = 0.5 * keep_ratio
    # Butterworth: 1 / (1 + (r / cutoff)^(2*order))
    mask = (1.0 / (1.0 + (r / cutoff).pow(2 * order))
            ).to(dtype).view(1, 1, X, Y)

    f_low = f * mask
    f_low = torch.fft.ifftshift(f_low, dim=(-2, -1))
    return torch.fft.ifft2(f_low, dim=(-2, -1)).real
