"""Lightweight stand-ins for predict.py's heavy runtime dependencies.

The production requirements.txt pulls a CUDA-specific torch build plus
whisperx, speechbrain, demucs, NeMo, google-cloud-pubsub, etc. -- gigabytes
of GPU-only dependencies that would make the test suite slow, flaky, and
impossible to run in ordinary CI. Only `torch` (CPU), `pandas`, `numpy`,
`requests` and `pytest` are real here (see requirements-test.txt); every
other import predict.py/nemotron_diarization.py touch at module load time is
registered as a minimal fake in sys.modules below, once, before any test
module imports predict/confirmation/nemotron_diarization.

This means these tests run against a deliberately separate, lightweight
environment -- not the same virtualenv you'd use to actually run `cog
predict`. Don't mix `pip install -r requirements-test.txt` with the real
requirements.txt in the same environment.
"""
import sys
import types


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _package_with_submodule(pkg_name, sub_name, **sub_attrs):
    pkg = _module(pkg_name)
    pkg.__path__ = []
    sub = _module(f"{pkg_name}.{sub_name}", **sub_attrs)
    setattr(pkg, sub_name, sub)
    return pkg, sub


def pytest_configure(config):
    if "predict" in sys.modules:
        return  # already stubbed (e.g. re-entrant test run in the same process)

    # --- cog ---
    def fake_input(*_args, default=None, **_kwargs):
        return default

    class FakeBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class FakeBasePredictor:
        pass

    _module("cog", Input=fake_input, BaseModel=FakeBaseModel, BasePredictor=FakeBasePredictor)

    # --- whisperx (+ .alignment submodule) ---
    _package_with_submodule(
        "whisperx",
        "alignment",
        DEFAULT_ALIGN_MODELS_TORCH={"es": "dummy-es-model"},
        DEFAULT_ALIGN_MODELS_HF={"en": "dummy-en-model"},
    )
    whisperx_mod = sys.modules["whisperx"]
    for name in ("load_model", "load_audio", "align", "load_align_model", "assign_word_speakers"):
        setattr(whisperx_mod, name, None)  # tests monkeypatch these explicitly before use

    # --- whisper.tokenizer ---
    _package_with_submodule(
        "whisper",
        "tokenizer",
        LANGUAGES={"en": "english", "es": "spanish"},
        TO_LANGUAGE_CODE={"english": "en", "spanish": "es"},
    )

    # --- nltk ---
    _module("nltk", download=lambda *a, **k: None)

    # --- google.cloud.pubsub_v1 / google.oauth2.service_account ---
    class FakePublisherClient:
        def __init__(self, *a, **k):
            pass

        def topic_path(self, *a, **k):
            return "fake-topic-path"

        def publish(self, *a, **k):
            class _Future:
                def result(self_inner):
                    return None
            return _Future()

    class FakeCredentials:
        @staticmethod
        def from_service_account_info(*a, **k):
            return object()

    google_mod = _module("google")
    google_mod.__path__ = []
    google_cloud_mod = _module("google.cloud")
    google_cloud_mod.__path__ = []
    google_mod.cloud = google_cloud_mod
    pubsub_v1_mod = _module("google.cloud.pubsub_v1", PublisherClient=FakePublisherClient)
    google_cloud_mod.pubsub_v1 = pubsub_v1_mod

    google_oauth2_mod = _module("google.oauth2")
    google_oauth2_mod.__path__ = []
    google_mod.oauth2 = google_oauth2_mod
    service_account_mod = _module("google.oauth2.service_account", Credentials=FakeCredentials)
    google_oauth2_mod.service_account = service_account_mod

    # --- pydub ---
    class FakeAudioSegment:
        def __init__(self, sample_width=2):
            self.sample_width = sample_width

        @classmethod
        def from_file(cls, *_a, **_k):
            return cls()

        def set_channels(self, _n):
            return self

        def __getitem__(self, _slice):
            return self

        def get_array_of_samples(self):
            return [0] * 100

    _module("pydub", AudioSegment=FakeAudioSegment)

    # --- speechbrain.pretrained.EncoderClassifier ---
    import numpy as np

    class FakeTensorChain:
        def __init__(self, values):
            self._values = values

        def squeeze(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array(self._values)

    class FakeEncoderClassifier:
        @classmethod
        def from_hparams(cls, *_a, **_k):
            return cls()

        def encode_batch(self, *_a, **_k):
            return FakeTensorChain([0.1, 0.2, 0.3])

    _package_with_submodule("speechbrain", "pretrained", EncoderClassifier=FakeEncoderClassifier)

    # --- transformers ---
    # Only needed as an importable placeholder: nemotron_diarization tests
    # monkeypatch its private _load() directly instead of calling these.
    _module("transformers", AutoModelForAudioFrameClassification=object, AutoProcessor=object)
