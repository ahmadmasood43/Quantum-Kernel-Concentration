import numpy as np
import pytest

from qkc.data import SYNTH_SEPARATION, make_synthetic, sha256, verify_sha256


def test_verify_sha256_accepts_match_and_rejects_mismatch(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"fashion")
    verify_sha256(p, sha256(p))
    with pytest.raises(ValueError):
        verify_sha256(p, "0" * 64)


def test_synthetic_default_separation_matches_experiments():
    a, ya = make_synthetic(seed=3, n_features=4)
    b, yb = make_synthetic(seed=3, n_features=4, separation=SYNTH_SEPARATION)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ya, yb)
    assert SYNTH_SEPARATION == 2.0
