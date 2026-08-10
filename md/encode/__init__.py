"""MuQ feature encoding: audio loading, the (chunked) MuQ forward, beat-grid alignment and the
resumable cache build. Ported from VFT vft/encode with two deltas: keys are music-id based (never
paths), and beat alignment is a read-time interpolation over the native 25Hz cache (see md.config)."""
