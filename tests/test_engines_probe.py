import sys
from types import SimpleNamespace

from app import engines


def _fake_ct2(monkeypatch, count=None, error=None):
    def get_count():
        if error is not None:
            raise error
        return count

    monkeypatch.setitem(
        sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=get_count)
    )


def _fake_torch(monkeypatch, hip=None, available=False, error=None):
    def is_available():
        if error is not None:
            raise error
        return available

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(hip=hip), cuda=SimpleNamespace(is_available=is_available)),
    )


def test_cpu_always_available(monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert engines.probe_engines()["cpu"]["available"] is True


def test_cuda_available_with_devices(monkeypatch):
    _fake_ct2(monkeypatch, count=2)
    monkeypatch.setitem(sys.modules, "torch", None)
    cuda = engines.probe_engines()["cuda"]
    assert cuda["available"] is True
    assert "2" in cuda["detail"]


def test_cuda_unavailable_zero_devices(monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    monkeypatch.setitem(sys.modules, "torch", None)
    cuda = engines.probe_engines()["cuda"]
    assert cuda["available"] is False
    assert cuda["detail"]


def test_cuda_probe_error_swallowed(monkeypatch):
    _fake_ct2(monkeypatch, error=RuntimeError("driver broken"))
    monkeypatch.setitem(sys.modules, "torch", None)
    cuda = engines.probe_engines()["cuda"]
    assert cuda["available"] is False
    assert "driver broken" in cuda["detail"]


def test_cuda_missing_ctranslate2(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    cuda = engines.probe_engines()["cuda"]
    assert cuda["available"] is False
    assert cuda["detail"]


def test_rocm_available(monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    _fake_torch(monkeypatch, hip="6.4.43484", available=True)
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is True
    assert "6.4.43484" in rocm["detail"]


def test_rocm_torch_not_installed(monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    monkeypatch.setitem(sys.modules, "torch", None)
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is False
    assert "未安装" in rocm["detail"]


def test_rocm_torch_without_hip(monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    _fake_torch(monkeypatch, hip=None, available=True)
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is False
    assert "非 ROCm" in rocm["detail"]


def test_rocm_hip_but_no_gpu(monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    _fake_torch(monkeypatch, hip="6.4.43484", available=False)
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is False


def test_rocm_probe_error_swallowed(monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    _fake_torch(monkeypatch, hip="6.4.43484", error=RuntimeError("hip init failed"))
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is False
    assert "hip init failed" in rocm["detail"]


def test_engines_endpoint(client, monkeypatch):
    _fake_ct2(monkeypatch, count=0)
    monkeypatch.setitem(sys.modules, "torch", None)
    resp = client.get("/api/engines")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"cpu", "cuda", "rocm"}
    assert data["cpu"]["available"] is True
    assert data["cuda"]["available"] is False
    assert data["rocm"]["available"] is False
    for item in data.values():
        assert isinstance(item["detail"], str) and item["detail"]