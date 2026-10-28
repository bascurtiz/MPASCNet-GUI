import torch
import torch.nn as nn
from models.mp_asc import MPASCEncoder, MPASCDecoder
from models.tsroformer import TSRoformer


class MPASCNet(nn.Module):
    def __init__(self, h):
        super(MPASCNet, self).__init__()
        self.h = h
        self.mp_asc_encoder = MPASCEncoder(h)
        self.tsroformer = TSRoformer(h)
        self.mp_asc_decoder = MPASCDecoder(h)

    def forward(self, noisy_mag, noisy_pha, progress_cb=None):
        if progress_cb:
            progress_cb(15, "Encoding magnitude & phase features...")
        feature = self.mp_asc_encoder(noisy_mag, noisy_pha)

        num_blocks = len(self.tsroformer.blocks)
        for i, block in enumerate(self.tsroformer.blocks):
            if progress_cb:
                pct = 20 + int((i / num_blocks) * 55)
                progress_cb(pct, f"Processing TSRoformer Layer {i + 1}/{num_blocks}...")
            feature = block(feature)

        if progress_cb:
            progress_cb(78, "Decoding high frequencies & neural phase...")
        denoised_mag, denoised_pha, denoised_com = self.mp_asc_decoder(feature, noisy_mag, noisy_pha)

        return denoised_mag, denoised_pha, denoised_com