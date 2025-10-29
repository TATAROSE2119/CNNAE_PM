import torch


class CombinedAE(torch.nn.Module):
    """
    Combine encoder (backbone) and decoder into a single module.
    forward returns (xhat, z) to be compatible with downstream utils.
    Input shape: [N, P, L]
    """
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        z = self.backbone(x)
        xhat = self.decoder(z)
        # Safely align the temporal length
        if xhat.size(-1) > x.size(-1):
            xhat = xhat[..., :x.size(-1)]
        elif xhat.size(-1) < x.size(-1):
            pad = x.size(-1) - xhat.size(-1)
            xhat = torch.nn.functional.pad(xhat, (0, pad))
        return xhat, z

