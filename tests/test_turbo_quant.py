"""Tests for TurboQuant compression."""

import numpy as np
import pytest

from mnemos.compression.turbo_quant import (
    TurboQuant,
    QuantizedTensor,
    quantize,
    dequantize,
    quantized_inner_product,
    _pack_codes,
    _unpack_codes,
    _scalar_quantize,
    _scalar_dequantize,
    _get_codebook,
)


class TestCodebook:
    def test_codebook_generation(self):
        centroids, boundaries = _get_codebook(128, 4)
        assert centroids.shape == (16,)
        assert boundaries.shape == (15,)
        assert np.all(centroids[:-1] < centroids[1:])

    def test_codebook_caching(self):
        c1, b1 = _get_codebook(128, 4)
        c2, b2 = _get_codebook(128, 4)
        assert c1 is c2


class TestBitPacking:
    def test_pack_unpack_4bit(self):
        codes = np.array([0, 1, 2, 15, 7, 3], dtype=np.uint8)
        packed = _pack_codes(codes, 4)
        recovered = _unpack_codes(packed, 4, len(codes))
        np.testing.assert_array_equal(codes, recovered)

    def test_pack_unpack_2bit(self):
        codes = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)
        packed = _pack_codes(codes, 2)
        recovered = _unpack_codes(packed, 2, len(codes))
        np.testing.assert_array_equal(codes, recovered)

    def test_pack_unpack_1bit(self):
        codes = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
        packed = _pack_codes(codes, 1)
        recovered = _unpack_codes(packed, 1, len(codes))
        np.testing.assert_array_equal(codes, recovered)


class TestTurboQuant:
    def test_mse_4bit_round_trip(self):
        rng = np.random.RandomState(42)
        vectors = rng.randn(10, 128).astype(np.float32)
        tq = TurboQuant(bits=4, mode="mse")
        qt = tq.quantize(vectors)
        recon = tq.dequantize(qt)
        assert recon.shape == vectors.shape
        mse = np.mean((vectors - recon) ** 2)
        assert mse < 0.5  # Reasonable reconstruction

    def test_prod_mode(self):
        vectors = np.random.randn(5, 128).astype(np.float32)
        tq = TurboQuant(bits=4, mode="prod")
        qt = tq.quantize(vectors)
        assert qt.qjl_bits is not None
        recon = tq.dequantize(qt)
        assert recon.shape == vectors.shape

    def test_inner_product(self):
        a = np.random.randn(3, 128).astype(np.float32)
        b = np.random.randn(4, 128).astype(np.float32)
        tq = TurboQuant(bits=4, mode="mse")
        qt_a = tq.quantize(a)
        qt_b = tq.quantize(b)
        result = tq.inner_product(qt_a, qt_b)
        assert result.shape == (3, 4)

    def test_compression_ratio(self):
        ratio = TurboQuant.compression_ratio((1000, 128), 4)
        assert ratio > 7.0  # 32/4 = 8x theoretical

    def test_convenience_api(self):
        vectors = np.random.randn(5, 128).astype(np.float32)
        qt = quantize(vectors, bits=4)
        recon = dequantize(qt)
        assert recon.shape == vectors.shape

    def test_single_vector(self):
        v = np.random.randn(128).astype(np.float32)
        tq = TurboQuant(bits=2)
        qt = tq.quantize(v)
        recon = tq.dequantize(qt)
        assert recon.shape == (1, 128)

    def test_save_load(self, tmp_path):
        vectors = np.random.randn(5, 128).astype(np.float32)
        tq = TurboQuant(bits=4, mode="mse")
        qt = tq.quantize(vectors)

        path = tmp_path / "test_qt.npz"
        tq.save(qt, path)
        qt_loaded = tq.load(path)

        assert qt_loaded.shape == qt.shape
        assert qt_loaded.bits == qt.bits
        np.testing.assert_array_equal(qt_loaded.codes, qt.codes)

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            TurboQuant(bits=0)
        with pytest.raises(ValueError):
            TurboQuant(bits=5)
