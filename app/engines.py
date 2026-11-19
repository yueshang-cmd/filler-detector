def probe_engines() -> dict:
    engines = {"cpu": {"available": True, "detail": "faster-whisper CPU int8 恒可用"}}
    try:
        import ctranslate2

        count = int(ctranslate2.get_cuda_device_count())
        engines["cuda"] = {
            "available": count > 0,
            "detail": (
                f"ctranslate2 检测到 {count} 个 CUDA 设备"
                if count > 0
                else "ctranslate2 未检测到 CUDA 设备"
            ),
        }
    except Exception as exc:
        engines["cuda"] = {"available": False, "detail": f"CUDA 探测失败：{exc}"}
    try:
        import torch

        hip = getattr(torch.version, "hip", None)
        if not hip:
            engines["rocm"] = {
                "available": False,
                "detail": "torch 非 ROCm 构建（torch.version.hip 为空）",
            }
        else:
            available = bool(torch.cuda.is_available())
            engines["rocm"] = {
                "available": available,
                "detail": f"torch ROCm {hip}，GPU 可用={available}",
            }
    except ImportError:
        engines["rocm"] = {
            "available": False,
            "detail": "torch 未安装（可选依赖见 requirements-rocm.txt）",
        }
    except Exception as exc:
        engines["rocm"] = {"available": False, "detail": f"ROCm 探测失败：{exc}"}
    return engines