"""Model-side tunables and encoder constants. Definitional constants of the chart domain live in
dio.common_struct; this file owns the audio/feature side. Values ported from VFT's audited config."""

MUQ_MODEL_NAME = "OpenMuQ/MuQ-large-msd-iter"          # the plain MuQ encoder, not MuQ-MuLan
MUQ_SAMPLE_RATE_HZ = 24_000                            # required input sample rate, mono
MUQ_FRAME_HZ = 25                                      # output frame rate: mel pooled to 25Hz -> 40ms per frame
MUQ_HOP_SAMPLES = MUQ_SAMPLE_RATE_HZ // MUQ_FRAME_HZ   # 960 input samples per output frame
MUQ_DIM = 1024                                         # hidden-state width
MUQ_N_LAYER = 13                                       # hidden_states count (embedding + 12 layers)

MUQ_MAX_SECONDS = 180                                  # single-pass ceiling; beyond it encode() chunks
MUQ_CHUNK_SECONDS = 150                                # chunk length for over-length audio
MUQ_CHUNK_OVERLAP_SECONDS = 20                         # chunk overlap; each chunk keeps its centre

STORE_DTYPE = "float16"                                # feature dtype on disk (MuQ infers in fp32)

GRID_STEPS_PER_BEAT = 12                               # model-facing feature frame rate: 1/12 beat per frame.
"""GRID_STEPS_PER_BEAT rationale: the cache stores NATIVE 25Hz frames (timing-independent asset);
beat alignment happens at read time by interpolation, so this constant is a free hyperparameter --
changing it re-reads, never re-encodes. 12 divides the 48-grid evenly (one feature frame per four
chart grid ticks) and sits near MuQ's native rate for typical BPM (150 bpm -> 30 frames/s)."""
