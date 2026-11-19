def test_settings_defaults_include_llm_and_watchdog(client):
    data = client.get("/api/settings").json()
    assert data["watchdog_seconds"] == 600
    assert data["llm"] == {"base_url": "", "key": "", "model": "", "enabled": False}


def test_llm_enabled_requires_base_url_and_model(client):
    resp = client.put("/api/settings", json={"llm": {"enabled": True}})
    assert resp.status_code == 400
    resp = client.put(
        "/api/settings", json={"llm": {"enabled": True, "base_url": "http://127.0.0.1:9/v1"}}
    )
    assert resp.status_code == 400
    resp = client.put(
        "/api/settings", json={"llm": {"enabled": True, "model": "x"}}
    )
    assert resp.status_code == 400


def test_llm_valid_config_persisted(client):
    cfg = {"enabled": True, "base_url": "http://127.0.0.1:9/v1", "key": "test", "model": "x"}
    resp = client.put("/api/settings", json={"llm": cfg})
    assert resp.status_code == 200
    assert client.get("/api/settings").json()["llm"] == cfg


def test_llm_partial_update_merges(client):
    cfg = {"enabled": True, "base_url": "http://127.0.0.1:9/v1", "key": "test", "model": "x"}
    assert client.put("/api/settings", json={"llm": cfg}).status_code == 200
    assert client.put("/api/settings", json={"llm": {"key": "k2"}}).status_code == 200
    got = client.get("/api/settings").json()["llm"]
    assert got["key"] == "k2"
    assert got["base_url"] == cfg["base_url"]
    assert got["enabled"] is True


def test_llm_enabled_must_be_bool(client):
    resp = client.put("/api/settings", json={"llm": {"enabled": "yes"}})
    assert resp.status_code == 400


def test_llm_must_be_object(client):
    assert client.put("/api/settings", json={"llm": "http://x"}).status_code == 400


def test_llm_fields_must_be_strings(client):
    resp = client.put("/api/settings", json={"llm": {"base_url": 123}})
    assert resp.status_code == 400


def test_llm_disabled_allows_empty_fields(client):
    resp = client.put("/api/settings", json={"llm": {"enabled": False}})
    assert resp.status_code == 200


def test_watchdog_seconds_validation(client):
    for bad in (0, -5, "600", True, None):
        assert client.put("/api/settings", json={"watchdog_seconds": bad}).status_code == 400
    for good in (1, 0.5, 600):
        assert client.put("/api/settings", json={"watchdog_seconds": good}).status_code == 200
    assert client.get("/api/settings").json()["watchdog_seconds"] == 600