"""Kafka is never on the training path (chap. 7.6) — proved, not just claimed.

The thesis argues that streaming serves inference and nothing else: training
reads files, deterministically, with no broker anywhere near it. That claim is
worth exactly as much as its enforcement, so this is an AST sweep rather than
a sentence in the documentation — it fails in CI the day someone imports
``kafka`` (or the serving package) from ``memoire.data``, ``memoire.model`` or
``memoire.training``.

The reverse direction is deliberately allowed: ``memoire.serving.inference``
imports the training code to rebuild the very model that was trained.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "memoire"
TRAINING_PACKAGES = ("data", "model", "training")
FORBIDDEN_PREFIXES = ("kafka", "memoire.serving")


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _python_files() -> list[Path]:
    return sorted(
        path
        for package in TRAINING_PACKAGES
        for path in (SRC / package).rglob("*.py")
    )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_training_path_never_imports_kafka_or_serving(path: Path) -> None:
    modules = _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
    offenders = sorted(
        module
        for module in modules
        if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_PREFIXES)
    )
    assert not offenders, f"{path} imports {offenders}: streaming must stay off the training path"


def test_the_sweep_actually_looks_at_something() -> None:
    """Guard against the sweep silently passing over an empty file list."""
    assert len(_python_files()) >= 10


def test_only_kafka_client_imports_kafka() -> None:
    """Inside the serving package too, kafka stays in one module."""
    for path in sorted((SRC / "serving").rglob("*.py")):
        modules = _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
        if path.name == "kafka_client.py":
            continue
        assert not [m for m in modules if m == "kafka" or m.startswith("kafka.")], (
            f"{path} imports kafka: adapters belong in memoire/serving/kafka_client.py"
        )


def test_the_service_loop_does_not_import_torch() -> None:
    """The loop is testable without torch: only the engine module needs it."""
    torch_free = ("service.py", "messages.py", "ports.py", "blobstore.py",
                  "preprocess.py", "producer.py", "kafka_client.py", "config.py")
    for name in torch_free:
        modules = _imported_modules(ast.parse((SRC / "serving" / name).read_text("utf-8")))
        assert not [m for m in modules if m == "torch" or m.startswith("torch.")], (
            f"memoire/serving/{name} imports torch"
        )
        assert "memoire.serving.inference" not in modules, (
            f"memoire/serving/{name} pulls the torch engine transitively"
        )


class _BlockKafka:
    """Import hook that makes ``import kafka`` fail, whatever is installed."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "kafka" or fullname.startswith("kafka."):
            raise ImportError("kafka-python is not installed (simulated)")
        # Anything else: fall through to the next finder (implicit None).


@pytest.fixture
def without_kafka(monkeypatch):
    monkeypatch.delitem(sys.modules, "kafka", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockKafka(), *sys.meta_path])
    yield


def test_kafka_client_imports_without_the_library(without_kafka) -> None:
    """The optional extra must not be needed to import — or to test — the
    package: ``kafka`` is only touched inside the factory functions."""
    module = importlib.reload(importlib.import_module("memoire.serving.kafka_client"))
    with pytest.raises(RuntimeError, match="kafka-python is not installed"):
        module.build_consumer("localhost:9092", "t", "g")
    with pytest.raises(RuntimeError, match=r"\[serve\]"):
        module.build_producer("localhost:9092")
