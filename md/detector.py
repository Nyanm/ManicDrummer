"""The detection-head baseline (f4): a small non-causal Transformer over beat-grid features,
per-tatum 9-lane onset + velocity + lane-8 pedal semantic. GOCT-scale on purpose (survey judgment:
capacity is not the bottleneck) -- this model exists to validate the data pipeline and set the M1
baseline numbers, not to be the final architecture.

Input [B, T, 13, 1024]: the 13 MuQ layers are mixed by a learnable softmax (VFT's audited form),
projected to d_model, plus a beat-phase embedding (tatum index mod TATUM_PER_BEAT) -- the beat-grid
prior as an input feature rather than an attention bias (that part waits for VFT4)."""
import torch
from torch import nn

from dio.common_struct import N_LANE

from .dataset import TATUM_PER_BEAT
from . import config

N_PEDAL_CLASS = 4  # PEDAL_NONE / HH / BD / UNKNOWN


class DrumDetector(nn.Module):
    def __init__(self, d_model: int = 256, n_layer: int = 3, n_head: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_logit = nn.Parameter(torch.zeros(config.MUQ_N_LAYER))
        self.project = nn.Linear(config.MUQ_DIM, d_model)
        self.embed_phase = nn.Embedding(TATUM_PER_BEAT, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout,
                                                   batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layer)
        self.head_onset = nn.Linear(d_model, N_LANE)
        self.head_velocity = nn.Linear(d_model, N_LANE)
        self.head_pedal = nn.Linear(d_model, N_PEDAL_CLASS)

    def forward(self, feat: torch.Tensor, tatum_lo: torch.Tensor) -> dict:
        """feat [B, T, n_layer, dim]; tatum_lo [B] window start tatum (for the absolute beat phase).
        Returns logits: onset [B,T,9], velocity [B,T,9] (sigmoid-scaled 0..1), pedal [B,T,4]."""
        mixed = torch.einsum("btld,l->btd", feat, torch.softmax(self.layer_logit, dim=0))
        arr_phase = (tatum_lo[:, None] + torch.arange(feat.shape[1], device=feat.device)[None, :]) % TATUM_PER_BEAT
        hidden = self.encoder(self.project(mixed) + self.embed_phase(arr_phase))
        return {
            "onset": self.head_onset(hidden),
            "velocity": torch.sigmoid(self.head_velocity(hidden)),
            "pedal": self.head_pedal(hidden),
        }


def phase_weight(arr_tatum: torch.Tensor) -> torch.Tensor:
    """ITGPT's beat-position BCE weighting adapted to the 24-tatum: straight subdivisions
    (beat/8th/16th) weigh 2.0, triplet-family 1.5, everything finer 0.5 -- the model must not buy
    accuracy on rare fine positions by spamming, and common positions carry the musical structure."""
    phase = arr_tatum % TATUM_PER_BEAT
    weight = torch.full_like(phase, 0.5, dtype=torch.float32)
    weight[(phase % 4 == 0)] = 1.5                    # triplet family (1/6, 1/12 beat) and their unions
    weight[(phase % 6 == 0)] = 2.0                    # beat / 8th / 16th
    return weight
