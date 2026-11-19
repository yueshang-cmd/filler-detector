"""M1 边界回归（测试工程师补充，仅测试代码，不改业务逻辑）。"""
import json
import os

import pytest

from app import settings
from tests.conftest import SAMPLE_MP4, wait_done


@pytest.mark.parametrize("name", ["virus.exe", "notes.txt", "noext", "archive.zip"])
def test_upload_unsupported_extension_rejected(client, name):
    before = client.get("/api/jobs").json()
    resp = client.post(
        "/api/jobs",
        files={"file": (name, b"whatever", "application/octet-stream")},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "不支持" in detail and "支持" in detail  # error 清晰、列出支持类型
    after = client.get("/api/jobs").json()
    assert len(after) == len(before)  # 不应起任务


def test_upload_corrupt_mp4_fails_gracefully(client):
    payload = os.urandom(64 * 1024)  # 随机字节假 mp4
    resp = client.post(
        "/api/jobs", files={"file": ("corrupt.mp4", payload, "video/mp4")}
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    data = wait_done(client, job_id)
    assert data["status"] == "failed", data
    assert data["error"] and data["error"].strip()  # error 非空
    assert data["result"] is None
    # 服务不崩：后续请求仍正常
    assert client.get("/api/settings").status_code == 200
    assert client.get("/api/jobs").status_code == 200


def test_get_job_not_found_variants(client):
    for jid in ["nonexistent", "000000000000", "..%2F..%2Fetc"]:
        assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_settings_put_engine_and_words_persist(client):
    payload = {
        "default_engine": "cuda",
        "default_model": "medium",
        "filler_words": {
            "pure": ["嗯", "呃", "um"],
            "contextual": ["这个", "然后", "like"],
        },
    }
    resp = client.put("/api/settings", json=payload)
    assert resp.status_code == 200
    got = client.get("/api/settings").json()
    assert got["default_engine"] == "cuda"
    assert got["default_model"] == "medium"
    assert got["filler_words"] == payload["filler_words"]
    # 落盘验证：settings.json 内容与读回一致
    on_disk = json.loads(settings.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert on_disk == got


def test_settings_put_invalid_words_rejected(client):
    resp = client.put(
        "/api/settings", json={"filler_words": {"pure": "嗯", "contextual": []}}
    )
    assert resp.status_code == 400
    # 非法请求不应污染已保存配置
    assert client.get("/api/settings").json()["filler_words"]["pure"][0] == "嗯"


def test_upload_uppercase_extension_accepted(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs", files={"file": ("SAMPLE.MP4", f, "video/mp4")}
        )
    assert resp.status_code == 200
    data = wait_done(client, resp.json()["job_id"])
    assert data["status"] == "done", data


def test_upload_traversal_filename_sanitized(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("../../evil.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 200
    data = wait_done(client, resp.json()["job_id"])
    assert data["status"] == "done", data
    assert data["filename"] == "evil.mp4"  # Path().name 去掉目录部分