"""render: audio reconstruction from game assets -- Konami ADPCM decoding, BMP beds, keysound
mixing and part-combination stems. Ported from iidxOnKnitting's audited Rust pipeline (m4f0-f4:
gain model calibrated to within 0.6 dB of the game's own masters); this is the key-sound dividend's
delivery channel: drum stems, no-drum accompaniments and per-note gain perturbations for training."""
