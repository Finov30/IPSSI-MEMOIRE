"""scripts/serve_inference.py's actual CLI (argparse -> wiring -> loop), the
surface a container runs — with the two Kafka adapters and the torch engine
swapped out, so the test needs no broker and no checkpoint.

Same loading pattern as tests/test_run_volume_curve_cli.py: the script is not
importable as a module, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from memoire.serving.config import load_streaming_config, resolve_streaming_config
from memoire.serving.ports import DamageInstance, InboundRecord, InferenceResult, ModelInfo
from memoire.serving.service import EXIT_TEMPFAIL

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_inference.py"
INSPECTION = "NCE01/AB-123-CD/2026-08-24T09:12:03Z"


def _load_script():
    spec = importlib.util.spec_from_file_location("serve_inference_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


class FakeSource:
    def __init__(self, records) -> None:
        self.batches = [list(records)]
        self.committed: list[int] = []
        self.closed = 0

    def poll(self, timeout_ms: int = 1000, max_records: int = 8):
        return self.batches.pop(0) if self.batches else []

    def commit(self, record) -> None:
        self.committed.append(record.offset)

    def close(self) -> None:
        self.closed += 1


class FakeSink:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes | None, bytes]] = []
        self.flushes = 0
        self.closed = 0

    def send(self, topic, key, value, headers=()) -> None:
        self.sent.append((topic, key, value))

    def flush(self, timeout_s: float = 30.0) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed += 1


class ConstantEngine:
    info = ModelInfo("unet", "binary", 64, 2, ("background", "damage"), "sha", "runs/dev", 10)

    def predict(self, image) -> InferenceResult:
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        return InferenceResult(mask, [DamageInstance(1, "damage", 0.9, (1, 1, 3, 3), 9)], 0.01, 5.0)


def _write_config(tmp_path: Path) -> Path:
    config = {
        "bootstrap_servers": "kafka:9092",
        "blob": {
            "photos_root": str(tmp_path / "photos"),
            "masks_root": str(tmp_path / "masks"),
            "scheme": "file",
        },
        "model": {"checkpoint": str(tmp_path / "best.pt"), "device": "cpu"},
        "retry": {"max_attempts": 1, "backoff_s": 0.0},
    }
    path = tmp_path / "streaming.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _photo_record(tmp_path: Path, offset: int = 0) -> InboundRecord:
    """A reference-payload message whose blob really sits under photos_root."""
    from memoire.serving import messages as msg
    from memoire.serving.blobstore import FilesystemBlobStore, sha256_hex

    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), (10, 200, 10)).save(buffer, format="PNG")
    data = buffer.getvalue()
    store = FilesystemBlobStore(tmp_path / "photos")
    uri = store.put("photos/p1.png", data, "image/png")
    body = msg.encode_photo(
        "p1", INSPECTION, "NCE01",
        msg.PhotoPayload("reference", "image/png", uri=uri, size_bytes=len(data),
                         sha256=sha256_hex(data)),
    )
    return InboundRecord("inspection.photos.v1", 0, offset, INSPECTION.encode(), body)


def _patch_transport(monkeypatch, script, source, sink) -> None:
    from memoire.serving import inference

    monkeypatch.setattr(inference, "load_engine", lambda *a, **k: ConstantEngine())
    monkeypatch.setattr(script, "build_consumer", lambda *a, **k: source)
    monkeypatch.setattr(script, "build_producer", lambda *a, **k: sink)
    monkeypatch.setattr(script, "KafkaMessageSource", lambda consumer: consumer)
    monkeypatch.setattr(script, "KafkaMessageSink", lambda producer: producer)
    monkeypatch.setattr(script, "install_signal_handlers", lambda: (lambda: False))


def test_cli_consumes_a_photo_and_publishes_a_mask(script, tmp_path, monkeypatch, capsys) -> None:
    config_path = _write_config(tmp_path)
    source = FakeSource([_photo_record(tmp_path)])
    sink = FakeSink()
    _patch_transport(monkeypatch, script, source, sink)

    code = script.main(["--config", str(config_path), "--max-iterations", "1"])

    assert code == 0
    topic, key, value = sink.sent[0]
    assert topic == "inspection.masks.v1"
    assert key == INSPECTION.encode()
    body = json.loads(value)
    assert body["mask"]["width"] == 16 and body["mask"]["height"] == 12
    assert source.committed == [0]
    assert json.loads(capsys.readouterr().out)["n_published"] == 1
    # The mask blob really landed under masks_root, next to nothing else.
    assert list((tmp_path / "masks").rglob("*.png"))


def test_cli_exits_75_when_the_failure_is_transient(script, tmp_path, monkeypatch) -> None:
    """EX_TEMPFAIL, not 0: the offset was not committed, and the restart must
    replay it rather than let the photo disappear."""
    config_path = _write_config(tmp_path)
    record = _photo_record(tmp_path, offset=4)
    (tmp_path / "photos").rename(tmp_path / "photos-gone")  # store unmounted

    class BrokenSink(FakeSink):
        def send(self, topic, key, value, headers=()) -> None:
            raise ConnectionError("broker down")

    source = FakeSource([record])
    sink = BrokenSink()
    _patch_transport(monkeypatch, script, source, sink)

    code = script.main(["--config", str(config_path), "--max-iterations", "1"])

    assert code == EXIT_TEMPFAIL
    assert source.committed == []


def test_cli_overrides_beat_the_config_file(script, tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path)
    source = FakeSource([])
    sink = FakeSink()
    seen: dict[str, object] = {}
    from memoire.serving import inference

    def _load_engine(checkpoint, **kwargs):
        seen["checkpoint"] = str(checkpoint)
        seen.update(kwargs)
        return ConstantEngine()

    def _build_consumer(servers, *args, **kwargs):
        seen["servers"] = servers
        return source

    monkeypatch.setattr(inference, "load_engine", _load_engine)
    monkeypatch.setattr(script, "build_consumer", _build_consumer)
    monkeypatch.setattr(script, "build_producer", lambda *a, **k: sink)
    monkeypatch.setattr(script, "KafkaMessageSource", lambda consumer: consumer)
    monkeypatch.setattr(script, "KafkaMessageSink", lambda producer: producer)
    monkeypatch.setattr(script, "install_signal_handlers", lambda: (lambda: False))

    script.main([
        "--config", str(config_path),
        "--checkpoint", "runs/other/best.pt",
        "--device", "cpu",
        "--bootstrap-servers", "localhost:29092",
        "--max-iterations", "1",
    ])

    assert seen["checkpoint"] == "runs/other/best.pt"
    assert seen["servers"] == "localhost:29092"


def test_resolve_streaming_config_defaults_and_rejects_typos() -> None:
    config = resolve_streaming_config({"group_id": "replay-1"})
    assert config["group_id"] == "replay-1"
    assert config["photos_topic"] == "inspection.photos.v1"
    # device stays cpu unless asked: "auto" would take the training GPU.
    assert config["model"]["device"] == "cpu"
    with pytest.raises(ValueError, match="unknown streaming config key"):
        resolve_streaming_config({"bootstrap_server": "kafka:9092"})


def test_shipped_config_file_is_valid() -> None:
    """configs/streaming.yaml must load with no unknown key — it is what the
    container starts with."""
    path = Path(__file__).resolve().parents[1] / "configs" / "streaming.yaml"
    config = load_streaming_config(path)
    assert config["masks_topic"] == "inspection.masks.v1"
    assert config["consumer"]["max_poll_records"] == 8
    assert config["model"]["device"] == "cpu"


def test_parse_args_defaults(script) -> None:
    args = script.parse_args([])
    assert args.config is None and args.checkpoint is None and args.max_iterations is None
