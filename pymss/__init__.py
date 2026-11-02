"""Public Python API for pymss.

pymss provides model catalog helpers, model downloading, audio I/O, ensemble
utilities, logging helpers, and the ``MSSeparator`` runtime for music source
separation. Most users can import from this top-level package instead of
importing submodules directly.

Exports:
    MSSeparator: Main runtime class for loading separation models and producing
        stems. Prefer ``MSSeparator.from_model_name(...)`` for catalog models.
    ModelTypeDetectionError: RuntimeError subclass for architecture detection failures.
    detect_model_type: Detect an architecture from a loaded YAML configuration.
    get_separation_logger: Create or reuse the package logger.
    create_separator: Create ``MSSeparator`` from a catalog model name.
    get_model_entry: Resolve catalog metadata for one model name or alias.
    list_models: List model catalog entries.
    list_user_models: List locally registered custom models.
    register_model: Register a local model path/config under a reusable name.
    unregister_model: Remove a previously registered custom model.
    resolve_model: Resolve catalog or user model paths without constructing a
        separator.
    download_model: Download all files required by one catalog model.
    ensemble_audios: Load and combine multiple audio files.
    save_ensemble_audio: Ensemble multiple audio files and save the result.
    WorkflowRunner: Run a multi-model audio workflow.
    load_audio: Load an audio file into a NumPy array.
    save_audio: Save a NumPy audio array to wav/flac/mp3/m4a.

Example:
    >>> from pymss import MSSeparator
    >>> separator = MSSeparator.from_model_name(
    ...     "bs_roformer_voc_hyperacev2",
    ...     download=True,
    ...     model_dir="models",
    ... )
    >>> separator.process_folder("song.wav")

Example:
    >>> from pymss import download_model, list_models
    >>> models = list_models(supported=True)
    >>> download_model(models[0].name, model_dir="models")

Example:
    >>> from pymss import ensemble_audios, save_ensemble_audio
    >>> audio, sample_rate = ensemble_audios(["a.wav", "b.wav"], weights=[1, 1])
    >>> save_ensemble_audio(["a.wav", "b.wav"], "ensemble.wav")
"""

from importlib import import_module as _import_module

# Keep package initialization light so `python -m pymss` can select its core
# before importing inference dependencies from the base installation.
_EXPORT_MODULES = {
    "MSSeparator": ".separator",
    "ModelTypeDetectionError": "pymss_core",
    "detect_model_type": "pymss_core",
    "get_separation_logger": ".logger",
    "create_separator": ".model_registry",
    "get_model_entry": ".model_registry",
    "list_models": ".model_registry",
    "register_model": ".model_registry",
    "resolve_model": ".model_registry",
    "unregister_model": ".model_registry",
    "ProxyError": ".model_download",
    "download_model": ".model_download",
    "list_user_models": ".user_models",
    "ensemble_audios": ".ensemble",
    "save_ensemble_audio": ".ensemble",
    "load_audio": ".audio_io",
    "save_audio": ".audio_io",
    "WorkflowRunner": ".workflow",
    "load_workflow_file": ".workflow",
    "run_workflow_file": ".workflow",
    "validate_workflow": ".workflow",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORT_MODULES.keys())


__all__ = (
    "MSSeparator",
    "ModelTypeDetectionError",
    "detect_model_type",
    "get_separation_logger",
    "create_separator",
    "get_model_entry",
    "list_models",
    "list_user_models",
    "register_model",
    "unregister_model",
    "resolve_model",
    "download_model",
    "ensemble_audios",
    "save_ensemble_audio",
    "WorkflowRunner",
    "load_workflow_file",
    "run_workflow_file",
    "validate_workflow",
    "load_audio",
    "save_audio",
)
