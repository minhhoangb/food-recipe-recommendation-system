import json
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from scripts import publish_artifacts


def test_release_uses_requested_metadata_and_uploads_checksum(tmp_path, monkeypatch, capsys):
    artifact = tmp_path / "model.zip.sha256"
    artifact.write_text("known-sha256  model.zip\n")
    notes = tmp_path / "notes.md"
    notes.write_text("Measured GPU training artifacts", encoding="utf-8")
    monkeypatch.setattr(publish_artifacts.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="username=fixture\npassword=fixture\n"))
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.get_method() == "GET":
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        if "uploads.example" in request.full_url:
            assert request.data == artifact.read_bytes()
            return BytesIO(json.dumps({"browser_download_url": "https://example.com/checksum"}).encode())
        payload = json.loads(request.data)
        assert payload == {"tag_name": "gpu-run", "name": "GPU artifacts",
                           "body": "Measured GPU training artifacts", "target_commitish": "abc123"}
        return BytesIO(json.dumps({"upload_url": "https://uploads.example/assets{?name}",
                                   "assets": [], "html_url": "https://example.com/release"}).encode())

    monkeypatch.setattr(publish_artifacts.urllib.request, "urlopen", urlopen)
    assert publish_artifacts.main([str(artifact), "--tag", "gpu-run", "--name", "GPU artifacts",
                                   "--notes-file", str(notes), "--target", "abc123"]) == 0
    assert len(requests) == 3
    output = json.loads(capsys.readouterr().out)
    assert output["url"] == "https://example.com/checksum"
    assert len(output["sha256"]) == 64


def test_missing_artifact_fails_before_reading_credentials(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Credentials must not be requested for a missing artifact")

    monkeypatch.setattr(publish_artifacts.subprocess, "run", unexpected)
    with pytest.raises(SystemExit) as error:
        publish_artifacts.main([str(tmp_path / "missing.zip")])
    assert error.value.code == 2
