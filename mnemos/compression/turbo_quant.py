"""
TurboQuant — Near-optimal Online Vector Quantization
=====================================================

Reference: arXiv:2504.19874
    "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
    Zandieh, Daliri, Hadian, Mirrokni (2025)

Algorithm summary:
    1. Apply a random rotation (seeded Hadamard or orthogonal) to the input
       vectors so that every coordinate follows a Beta(d/2, d/2) distribution.
    2. Quantize each coordinate independently using a precomputed
       Lloyd-Max codebook optimised for that Beta distribution.
    3. For inner-product mode, compose the MSE quantizer (at b-1 bits)
       with a 1-bit QJL (Quantized Johnson–Lindenstrauss) on the residual
       to obtain an unbiased inner-product estimator.

Bit-width / MSE bounds (unit-norm vectors):
    1-bit  →  MSE ≤ 0.36
    2-bit  →  MSE ≤ 0.117
    3-bit  →  MSE ≤ 0.03
    4-bit  →  MSE ≤ 0.009

Cross-domain note:
    This implementation compresses **embedding vectors** for storage in
    Qdrant / pgvector.  The same TurboQuant algorithm has been independently
    validated for **LLM KV cache compression** in llama.cpp
    (github.com/TheTom/turboquant_plus).  That application uses C/Metal/CUDA
    kernels and operates at inference time — it is architecturally separate
    from this module.  See docs/whitepaper.md §4.3 for details.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QuantizedTensor:
    """Packed quantized representation of a float32 tensor."""

    codes: np.ndarray          # uint8 packed codes — shape depends on packing
    shape: Tuple[int, ...]     # original float32 shape
    bits: int                  # quantization bit-width (1–4)
    mode: str                  # "mse" or "prod"
    rotation_seed: int         # seed used for the random rotation
    norms: np.ndarray          # per-vector L2 norms  (N,)
    codebook: np.ndarray       # 1-D centroids array  (2^b,)
    boundaries: np.ndarray     # decision boundaries   (2^b - 1,)

    # Only present in "prod" mode
    qjl_bits: Optional[np.ndarray] = None   # packed 1-bit QJL of residual
    qjl_seed: Optional[int] = None

    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lloyd-Max codebook computation
# ---------------------------------------------------------------------------

def _lloyd_max_beta(bits: int, alpha: float, beta_param: float,
                    n_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal Lloyd-Max codebook for Beta(alpha, beta_param) on [0, 1].

    Returns (centroids, boundaries) where
        centroids.shape  = (2^bits,)
        boundaries.shape = (2^bits - 1,)
    """
    from scipy import stats as sp_stats

    k = 1 << bits
    dist = sp_stats.beta(alpha, beta_param)

    # Initialise with uniform quantiles
    quantiles = np.linspace(0, 1, k + 1)
    boundaries = dist.ppf(quantiles[1:-1])           # (k-1,)
    centroids = np.zeros(k)

    for _ in range(n_iter):
        # ---- Centroid update (conditional expectation in each bin) ----
        edges = np.concatenate([[0.0], boundaries, [1.0]])
        for j in range(k):
            lo, hi = edges[j], edges[j + 1]
            if hi - lo < 1e-15:
                centroids[j] = (lo + hi) / 2
                continue
            # E[X | lo < X < hi]  via truncated moments
            p = dist.cdf(hi) - dist.cdf(lo)
            if p < 1e-15:
                centroids[j] = (lo + hi) / 2
                continue
            # Numerical integration for the conditional mean
            from scipy.integrate import quad
            num, _ = quad(lambda x: x * dist.pdf(x), lo, hi)
            centroids[j] = num / p

        # ---- Boundary update (midpoints of adjacent centroids) ----
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    return centroids, boundaries


def _precompute_codebooks(dimensions: tuple = (128, 384, 768, 1536),
                          max_bits: int = 4
                          ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Precompute Lloyd-Max codebooks for several (dim, bit-width) pairs.

    After random rotation the coordinates of a d-dimensional unit vector
    follow Beta(d/2, d/2) (shifted & scaled from [-1/√d, 1/√d] to [0,1]).
    """
    codebooks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for d in dimensions:
        alpha = d / 2.0
        for b in range(1, max_bits + 1):
            key = f"d{d}_b{b}"
            centroids, boundaries = _lloyd_max_beta(b, alpha, alpha)
            codebooks[key] = (centroids, boundaries)
    return codebooks


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _random_rotation_matrix(d: int, seed: int) -> np.ndarray:
    """
    Generate a d×d random orthogonal matrix via QR of a Gaussian matrix.
    Deterministic for a given seed.
    """
    rng = np.random.RandomState(seed)
    z = rng.randn(d, d)
    q, r = np.linalg.qr(z)
    # Make QR decomposition unique (Haar-distributed)
    sign = np.sign(np.diag(r))
    sign[sign == 0] = 1
    q *= sign[np.newaxis, :]
    return q.astype(np.float32)


def _fast_random_rotate(vectors: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply a seeded random orthogonal rotation.
    For small d we use an explicit Q matrix; for large d a randomised
    Hadamard would be faster but we keep it simple here.
    """
    d = vectors.shape[-1]
    Q = _random_rotation_matrix(d, seed)
    return vectors @ Q  # (N, d) @ (d, d) → (N, d)


def _fast_random_unrotate(vectors: np.ndarray, seed: int) -> np.ndarray:
    """Inverse rotation (Q is orthogonal so Q^{-1} = Q^T)."""
    d = vectors.shape[-1]
    Q = _random_rotation_matrix(d, seed)
    return vectors @ Q.T


# ---------------------------------------------------------------------------
# Scalar quantize / dequantize
# ---------------------------------------------------------------------------

def _scalar_quantize(values: np.ndarray,
                     boundaries: np.ndarray) -> np.ndarray:
    """Map continuous values to bin indices using decision boundaries."""
    return np.searchsorted(boundaries, values).astype(np.uint8)


def _scalar_dequantize(codes: np.ndarray,
                       centroids: np.ndarray) -> np.ndarray:
    """Map bin indices back to centroid values."""
    return centroids[codes]


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """
    Pack an array of uint8 codes (each in 0..2^bits-1) into a compact
    uint8 byte stream.
    """
    if bits == 8:
        return codes.copy()

    flat = codes.ravel()
    n = len(flat)
    codes_per_byte = 8 // bits

    # Pad to full byte boundary
    pad_len = (-n) % codes_per_byte
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])

    n_padded = len(flat)
    packed = np.zeros(n_padded // codes_per_byte, dtype=np.uint8)

    for i in range(codes_per_byte):
        packed |= flat[i::codes_per_byte].astype(np.uint8) << (i * bits)

    return packed


def _unpack_codes(packed: np.ndarray, bits: int, n_elements: int) -> np.ndarray:
    """Unpack a compact byte stream back to uint8 codes."""
    if bits == 8:
        return packed[:n_elements].copy()

    codes_per_byte = 8 // bits
    mask = (1 << bits) - 1

    total_unpacked = len(packed) * codes_per_byte
    out = np.zeros(total_unpacked, dtype=np.uint8)

    for i in range(codes_per_byte):
        out[i::codes_per_byte] = (packed >> (i * bits)) & mask

    return out[:n_elements]


# ---------------------------------------------------------------------------
# QJL (1-bit Quantized Johnson–Lindenstrauss) for residual
# ---------------------------------------------------------------------------

def _qjl_encode(residual: np.ndarray, seed: int) -> np.ndarray:
    """
    1-bit sign quantization of the residual after a random projection.
    Each coordinate → 1 bit (sign).  Packed into uint8.
    """
    # Simple sign-based QJL: apply random sign flips then take sign
    rng = np.random.RandomState(seed)
    d = residual.shape[-1]
    signs = rng.choice([-1, 1], size=d).astype(np.float32)
    projected = residual * signs  # random sign projection
    bits = (projected >= 0).astype(np.uint8)
    return _pack_codes(bits.ravel(), 1)


def _qjl_decode(packed_bits: np.ndarray, n_elements: int,
                seed: int) -> np.ndarray:
    """
    Dequantize 1-bit QJL codes.
    Returns ±1/√d scaled vectors.
    """
    bits = _unpack_codes(packed_bits, 1, n_elements)
    values = bits.astype(np.float32) * 2 - 1  # map {0,1} → {-1,+1}
    # Undo the random sign flips
    rng = np.random.RandomState(seed)
    d = n_elements  # last-dim size (works for flat case)
    signs = rng.choice([-1, 1], size=d).astype(np.float32)
    return values * signs


# ---------------------------------------------------------------------------
# TurboQuant class
# ---------------------------------------------------------------------------

_CACHED_CODEBOOKS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def _get_codebook(dim: int, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get or compute codebook for given (dim, bits)."""
    key = f"d{dim}_b{bits}"
    if key not in _CACHED_CODEBOOKS:
        alpha = dim / 2.0
        centroids, boundaries = _lloyd_max_beta(bits, alpha, alpha)
        _CACHED_CODEBOOKS[key] = (centroids, boundaries)
    return _CACHED_CODEBOOKS[key]


class TurboQuant:
    """
    TurboQuant quantizer with MSE and inner-product modes.

    Usage:
        tq = TurboQuant(bits=4, mode="mse")
        qt = tq.quantize(vectors)         # → QuantizedTensor
        recon = tq.dequantize(qt)          # → np.ndarray
        sim = tq.inner_product(qt_a, qt_b) # → float / array
    """

    def __init__(self, bits: int = 4,
                 mode: Literal["mse", "prod"] = "mse",
                 seed: int = 42):
        """
        Args:
            bits: Quantization bit-width (1–4).
            mode: "mse" for MSE-optimal, "prod" for unbiased inner-product.
            seed: Random seed for the rotation matrix.
        """
        if bits < 1 or bits > 4:
            raise ValueError(f"bits must be 1–4, got {bits}")
        if mode not in ("mse", "prod"):
            raise ValueError(f"mode must be 'mse' or 'prod', got {mode!r}")

        self.bits = bits
        self.mode = mode
        self.seed = seed

        logger.info(f"⚡ TurboQuant initialised: {bits}-bit, mode={mode}")

    # ──────────────────────── quantize ────────────────────────

    def quantize(self, vectors: np.ndarray) -> QuantizedTensor:
        """
        Quantize a batch of vectors.

        Args:
            vectors: float32 array of shape (N, D) or (D,).

        Returns:
            QuantizedTensor with packed codes and metadata.
        """
        t0 = time.time()
        single = vectors.ndim == 1
        if single:
            vectors = vectors[np.newaxis, :]

        N, D = vectors.shape

        # 1. Store and remove norms
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-10)
        unit_vectors = vectors / safe_norms

        # 2. Random rotation → Beta-distributed coordinates
        rotated = _fast_random_rotate(unit_vectors, self.seed)

        # 3. Shift from [-1,1] to [0,1] range for Beta quantizer
        scaled = (rotated + 1.0) / 2.0
        scaled = np.clip(scaled, 0.0, 1.0)

        # 4. Get codebook
        centroids, boundaries = _get_codebook(D, self._mse_bits)

        # 5. Scalar quantize each coordinate
        codes = _scalar_quantize(scaled.ravel(), boundaries)
        packed = _pack_codes(codes, self._mse_bits)

        # 6. QJL on residual (prod mode only)
        qjl_packed = None
        qjl_seed = None
        if self.mode == "prod":
            # Reconstruct from MSE quantizer
            recon_scaled = _scalar_dequantize(codes, centroids).reshape(N, D)
            recon_rotated = recon_scaled * 2.0 - 1.0
            residual = rotated - recon_rotated  # in rotated space
            qjl_seed = self.seed + 1_000_000
            qjl_packed = _qjl_encode(residual, qjl_seed)

        elapsed = time.time() - t0
        logger.debug(f"⚡ Quantized {N}×{D} → {self.bits}-bit in {elapsed:.3f}s")

        return QuantizedTensor(
            codes=packed,
            shape=vectors.shape,
            bits=self.bits,
            mode=self.mode,
            rotation_seed=self.seed,
            norms=norms.ravel(),
            codebook=centroids,
            boundaries=boundaries,
            qjl_bits=qjl_packed,
            qjl_seed=qjl_seed,
            metadata={
                "encoding_time_s": elapsed,
                "original_dtype": str(vectors.dtype),
            },
        )

    # ──────────────────────── dequantize ──────────────────────

    def dequantize(self, qt: QuantizedTensor) -> np.ndarray:
        """Reconstruct float32 vectors from a QuantizedTensor."""
        N, D = qt.shape
        n_elements = N * D

        codes = _unpack_codes(qt.codes, self._mse_bits, n_elements)
        recon_scaled = _scalar_dequantize(codes, qt.codebook).reshape(N, D)

        # Undo [0,1] scaling
        recon_rotated = recon_scaled * 2.0 - 1.0

        # Add QJL residual correction in prod mode
        if qt.mode == "prod" and qt.qjl_bits is not None:
            qjl_recon = _qjl_decode(qt.qjl_bits, n_elements, qt.qjl_seed)
            # Scale residual estimate; QJL gives ±1, scale by expected residual magnitude
            mse_bound = self._theoretical_mse(self._mse_bits)
            residual_scale = np.sqrt(mse_bound) if mse_bound > 0 else 0.01
            recon_rotated = recon_rotated + qjl_recon.reshape(N, D) * residual_scale

        # Inverse rotation
        recon_unit = _fast_random_unrotate(recon_rotated, qt.rotation_seed)

        # Re-apply norms
        recon = recon_unit * qt.norms[:, np.newaxis]
        return recon.astype(np.float32)

    # ──────────────────────── inner product ───────────────────

    def inner_product(self, qt_a: QuantizedTensor,
                      qt_b: QuantizedTensor) -> np.ndarray:
        """
        Approximate inner product between two quantized tensors.

        Returns:
            Array of shape (Na, Nb) with pairwise inner products.
        """
        # Reconstruct and compute (fast path via dequantize for now;
        # a production implementation would operate directly on codes)
        recon_a = self.dequantize(qt_a)
        recon_b = self.dequantize(qt_b)
        return recon_a @ recon_b.T

    # ──────────────────────── helpers ─────────────────────────

    @property
    def _mse_bits(self) -> int:
        """Bit-width used for the MSE quantizer stage."""
        if self.mode == "prod":
            return max(1, self.bits - 1)
        return self.bits

    @staticmethod
    def _theoretical_mse(bits: int) -> float:
        """Upper-bound MSE for unit-norm vectors at given bit-width."""
        bounds = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
        return bounds.get(bits, np.sqrt(3) * np.pi / 2 * (1 / 4 ** bits))

    # ──────────────────────── serialisation ───────────────────

    def save(self, qt: QuantizedTensor, path: Union[str, Path]) -> None:
        """Save a QuantizedTensor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "codes": qt.codes,
            "shape": np.array(qt.shape),
            "bits": np.array(qt.bits),
            "mode": np.array(qt.mode),
            "rotation_seed": np.array(qt.rotation_seed),
            "norms": qt.norms,
            "codebook": qt.codebook,
            "boundaries": qt.boundaries,
        }
        if qt.qjl_bits is not None:
            data["qjl_bits"] = qt.qjl_bits
            data["qjl_seed"] = np.array(qt.qjl_seed)

        np.savez_compressed(path, **data)
        logger.debug(f"💾 Saved QuantizedTensor to {path}")

    def load(self, path: Union[str, Path]) -> QuantizedTensor:
        """Load a QuantizedTensor from disk."""
        data = np.load(path, allow_pickle=True)

        qjl_bits = data["qjl_bits"] if "qjl_bits" in data else None
        qjl_seed = int(data["qjl_seed"]) if "qjl_seed" in data else None

        return QuantizedTensor(
            codes=data["codes"],
            shape=tuple(data["shape"]),
            bits=int(data["bits"]),
            mode=str(data["mode"]),
            rotation_seed=int(data["rotation_seed"]),
            norms=data["norms"],
            codebook=data["codebook"],
            boundaries=data["boundaries"],
            qjl_bits=qjl_bits,
            qjl_seed=qjl_seed,
        )

    # ──────────────────────── compression ratio ──────────────

    @staticmethod
    def compression_ratio(original_shape: Tuple[int, ...],
                          bits: int) -> float:
        """
        Theoretical compression ratio vs float32 storage.
        """
        n_elements = 1
        for s in original_shape:
            n_elements *= s
        original_bytes = n_elements * 4  # float32
        quantized_bits = n_elements * bits
        quantized_bytes = (quantized_bits + 7) // 8
        return original_bytes / max(quantized_bytes, 1)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

def quantize(vectors: np.ndarray, bits: int = 4,
             mode: str = "mse", seed: int = 42) -> QuantizedTensor:
    """Convenience: quantize vectors in one call."""
    tq = TurboQuant(bits=bits, mode=mode, seed=seed)
    return tq.quantize(vectors)


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """Convenience: dequantize a QuantizedTensor."""
    tq = TurboQuant(bits=qt.bits, mode=qt.mode, seed=qt.rotation_seed)
    return tq.dequantize(qt)


def quantized_inner_product(qt_a: QuantizedTensor,
                            qt_b: QuantizedTensor) -> np.ndarray:
    """Convenience: approximate inner product between quantized tensors."""
    tq = TurboQuant(bits=qt_a.bits, mode=qt_a.mode, seed=qt_a.rotation_seed)
    return tq.inner_product(qt_a, qt_b)
