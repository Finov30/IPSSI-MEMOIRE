"""Streaming inference path: photos in on Kafka, masks out (chap. 5.4).

Layering, from the inside out — each layer only knows the one below:

- ``ports``       : protocols and result types (no kafka, no torch);
- ``preprocess``  : letterbox of one arbitrary image (PIL/numpy only);
- ``messages``    : the JSON wire format of the three topics (pure);
- ``blobstore``   : the claim-check object store behind the references;
- ``inference``   : the torch engine rebuilt from a checkpoint;
- ``service``     : the consumer loop (offsets, DLQ, clean shutdown);
- ``producer``    : the upstream side, for replay and demonstration;
- ``kafka_client``: the kafka adapters, the only module importing kafka.

Nothing here is imported by ``memoire.training`` or ``memoire.model``:
Kafka is never on the training path (chap. 7.6), and
``tests/test_serving_isolation.py`` enforces it.

Only ``ports`` is re-exported: importing this package must stay cheap and must
not pull torch (``memoire.serving.inference``) or kafka.
"""

from memoire.serving.ports import (
    DamageInstance,
    InboundRecord,
    InferenceResult,
    MessageSink,
    MessageSource,
    ModelInfo,
    SegmentationEngine,
)

__all__ = [
    "DamageInstance",
    "InboundRecord",
    "InferenceResult",
    "MessageSink",
    "MessageSource",
    "ModelInfo",
    "SegmentationEngine",
]
