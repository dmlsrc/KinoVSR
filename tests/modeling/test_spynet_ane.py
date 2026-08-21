"""SpyNet on the Neural Engine: parity with the MLX reference, and the
fallbacks that keep a run working when that path is unavailable."""

from __future__ import annotations

import mlx.core as mx
import pytest

from kinovsr.modeling import spynet_ane
from kinovsr.modeling.vsr_blocks import (
    compiled_spynet_flow,
    mlx_spynet_flow,
    spynet_flow,
)

WEIGHTS = ("kinovsr/modeling/spynet/weights/spynet_stock_20210409.safetensors")


@pytest.fixture(scope="module")
def params():
    from pathlib import Path

    import kinovsr.modeling as modeling

    path = (Path(modeling.__file__).parent / "spynet" / "weights"
            / "spynet_stock_20210409.safetensors")
    return {k: v.astype(mx.float32) for k, v in mx.load(str(path)).items()}


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Each test gets its own converted-model cache and a clean verdict."""
    monkeypatch.setenv("KINOVSR_CACHE_DIR", str(tmp_path / "cache"))
    from kinovsr.settings import _reset_default_settings as reset_settings

    reset_settings()
    spynet_ane.reset_cache()
    yield
    spynet_ane.reset_cache()
    reset_settings()


def _wrong_input_width(params):
    """A noncanonical dict whose first convolution was padded from 8 to 16."""
    bad = dict(params)
    k = "spynet.basic_module.0.basic_module.0.conv.weight"
    w = bad[k]
    bad[k] = mx.concatenate(
        [w, mx.zeros((*w.shape[:3], 8), dtype=w.dtype)], axis=-1)
    return bad


def _pair(h=96, w=128, shift=3):
    ys, xs = mx.meshgrid(mx.arange(h), mx.arange(w + shift), indexing="ij")
    xf, yf = xs.astype(mx.float32), ys.astype(mx.float32)
    n = mx.sin(xf * 12.9898 + yf * 78.233) * 43758.5453
    tex = n - mx.floor(n)
    sheet = mx.stack([tex, tex * 0.8 + 0.1, 1.0 - tex * 0.6], axis=-1)
    a = mx.contiguous(sheet[:, :w])[None]
    b = mx.contiguous(sheet[:, shift:shift + w])[None]
    mx.eval(a, b)
    return a, b


class TestGeometry:
    def test_padding_and_levels_match_the_reference_rule(self):
        assert spynet_ane.padded_geometry(352, 640) == (352, 640, 6)
        assert spynet_ane.padded_geometry(300, 500) == (320, 512, 6)
        assert spynet_ane.padded_geometry(30, 30) == (32, 32, 5)

    def test_batched_and_tiny_inputs_are_declined(self, params):
        assert spynet_ane.engine_for(params, (2, 96, 128, 3)) is None
        assert spynet_ane.engine_for(params, (1, 8, 8, 3)) is None


class TestFailureScoping:
    """A failed conversion is remembered for its own key only. This
    regressed when one malformed weight dictionary made the then-global
    verdict blacklist every engine for the rest of the process."""

    def test_a_broken_key_does_not_poison_the_availability_verdict(
            self, params):
        if spynet_ane.unavailable_reason():
            pytest.skip(spynet_ane.unavailable_reason())
        assert spynet_ane.engine_for(_wrong_input_width(params), (1, 96, 128, 3)) is None
        assert spynet_ane.unavailable_reason() is None
        assert "expected 8 input channels, got 16" in (spynet_ane.last_failure() or "")


class TestBackendSelection:
    def test_mlx_backend_never_builds_an_engine(self, params, monkeypatch):
        monkeypatch.setenv("SPYNET_BACKEND", "mlx")
        from kinovsr.settings import _reset_default_settings as reset_settings

        reset_settings()

        def explode(*args, **kwargs):  # must not be consulted at all
            raise AssertionError("ANE engine built for spynet_backend=mlx")

        monkeypatch.setattr(spynet_ane, "engine_for", explode)
        a, b = _pair()
        got = compiled_spynet_flow(params, a, b)
        want = mlx_spynet_flow(params, a, b)
        mx.eval(got, want)
        assert float(mx.max(mx.abs(got - want))) == 0.0

    def test_ane_backend_raises_instead_of_falling_back(self, params,
                                                        monkeypatch):
        monkeypatch.setenv("SPYNET_BACKEND", "ane")
        from kinovsr.settings import _reset_default_settings as reset_settings

        reset_settings()
        monkeypatch.setattr(spynet_ane, "engine_for", lambda *a, **k: None)
        monkeypatch.setattr(spynet_ane, "unavailable_reason",
                            lambda: "forced for the test")
        a, b = _pair()
        with pytest.raises(RuntimeError, match="forced for the test"):
            compiled_spynet_flow(params, a, b)

    def test_auto_falls_back_when_the_engine_is_unavailable(self, params,
                                                            monkeypatch):
        monkeypatch.setattr(spynet_ane, "engine_for", lambda *a, **k: None)
        a, b = _pair()
        got = compiled_spynet_flow(params, a, b)
        want = mlx_spynet_flow(params, a, b)
        mx.eval(got, want)
        assert float(mx.max(mx.abs(got - want))) == 0.0


@pytest.mark.integration
class TestAneParity:
    def _engine(self, params, shape):
        engine = spynet_ane.engine_for(params, shape)
        if engine is None:
            pytest.skip(spynet_ane.unavailable_reason() or "ANE unavailable")
        return engine

    @pytest.mark.parametrize("h,w", [(96, 128), (352, 640)])
    def test_matches_the_mlx_reference(self, params, h, w):
        a, b = _pair(h, w)
        engine = self._engine(params, a.shape)
        got = engine.flow(a, b)
        want = spynet_flow(params, a, b)
        mx.eval(got, want)
        assert got.shape == want.shape
        epe = mx.sqrt(mx.sum((got - want) ** 2, axis=-1))
        assert float(mx.mean(epe)) < 0.02, float(mx.mean(epe))

    def test_unpadded_geometry_is_resized_and_rescaled(self, params):
        a, b = _pair(100, 140)          # neither dimension divides by 32
        engine = self._engine(params, a.shape)
        got = engine.flow(a, b)
        want = spynet_flow(params, a, b)
        mx.eval(got, want)
        assert got.shape == want.shape == (1, 100, 140, 2)
        epe = mx.sqrt(mx.sum((got - want) ** 2, axis=-1))
        assert float(mx.mean(epe)) < 0.05, float(mx.mean(epe))

    def test_repeated_calls_are_stable(self, params):
        """The output backings are reused across calls; a stale reference
        would show up as drift between identical invocations."""
        a, b = _pair()
        engine = self._engine(params, a.shape)
        first = engine.flow(a, b)
        mx.eval(first)
        first = mx.contiguous(first)
        for _ in range(3):
            again = engine.flow(a, b)
            mx.eval(again)
            assert float(mx.max(mx.abs(first - again))) == 0.0

    def test_one_broken_key_leaves_other_engines_live(self, params):
        a, b = _pair()
        engine = self._engine(params, a.shape)
        assert spynet_ane.engine_for(_wrong_input_width(params), a.shape) is None
        assert spynet_ane.engine_for(params, a.shape) is engine

    def test_conversion_is_cached_on_disk(self, params, tmp_path):
        a, _ = _pair()
        self._engine(params, a.shape)
        cached = list((tmp_path / "cache" / "spynet-ane").glob("*/level*.mlpackage"))
        assert len(cached) >= 5
        assert not list((tmp_path / "cache").rglob("*.partial.mlpackage"))


class TestCliOverrideReachesTheBackend:
    """The CLI resolves its own Settings; a flag that never reaches the
    engine silently runs the default backend (this regressed once)."""

    def test_flag_is_published_to_the_process_default(self, monkeypatch):
        from kinovsr.cli.args import build_parser
        from kinovsr.cli.config import assemble
        from kinovsr.settings import default_settings

        monkeypatch.delenv("SPYNET_BACKEND", raising=False)
        args = build_parser().parse_args(
            ["--video", "clip.mp4", "--spynet-backend", "mlx"])
        invocation = assemble(args)
        assert invocation.settings.spynet_backend == "mlx"
        assert default_settings().spynet_backend == "mlx"

    def test_default_is_auto(self, monkeypatch):
        from kinovsr.cli.args import build_parser
        from kinovsr.cli.config import assemble

        monkeypatch.delenv("SPYNET_BACKEND", raising=False)
        args = build_parser().parse_args(["--video", "clip.mp4"])
        assert assemble(args).settings.spynet_backend == "auto"


@pytest.mark.integration
class TestWarmStartCost:
    """Inference must not need coremltools, and must not recompile models
    that were already compiled - both regressed once and cost ~1.5 s per run."""

    def test_warm_start_skips_coremltools_and_reuses_compiled_models(
            self, params, tmp_path):
        import subprocess
        import sys

        cache = tmp_path / "cache"
        script = (
            "import warnings, sys, time; warnings.filterwarnings('ignore')\n"
            "import mlx.core as mx\n"
            "from pathlib import Path\n"
            "import kinovsr.modeling as M\n"
            "from kinovsr.modeling import spynet_ane\n"
            "w = Path(M.__file__).parent/'spynet'/'weights'/"
            "'spynet_stock_20210409.safetensors'\n"
            "p = {k: v.astype(mx.float32) for k, v in mx.load(str(w)).items()}\n"
            "eng = spynet_ane.engine_for(p, (1, 352, 640, 3))\n"
            "print('OK' if eng is not None else 'NONE',"
            " 'coremltools' in sys.modules)\n"
        )
        env = {**dict(__import__("os").environ), "KINOVSR_CACHE_DIR": str(cache)}
        first = subprocess.run([sys.executable, "-c", script], env=env,
                               capture_output=True, text=True, timeout=600)
        if "OK" not in first.stdout:
            pytest.skip(f"ANE unavailable: {first.stdout} {first.stderr[-300:]}")
        # The cold run converts and compiles; both artifacts must persist.
        geometry = next((cache / "spynet-ane").iterdir())
        assert (geometry / "level0.mlpackage").exists()
        assert (geometry / "level0.mlmodelc").exists()

        second = subprocess.run([sys.executable, "-c", script], env=env,
                                capture_output=True, text=True, timeout=600)
        assert "OK" in second.stdout, second.stderr[-300:]
        assert second.stdout.split()[1] == "False", (
            "warm start imported coremltools; inference must not need it")
