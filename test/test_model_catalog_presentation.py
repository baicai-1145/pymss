"""Check catalog presentation changes without changing model identity or storage."""

from pymss.model_registry import ModelEntry, get_model_entry, list_models


def test_sort_key_is_optional_for_older_catalogs():
    entry = ModelEntry.from_dict({"name": "custom.ckpt", "relpath": "custom.ckpt"})
    assert entry.sort_key == ""


def test_mega_family_orders_full_model_before_single_target_models():
    family = list_models(category="music_stems/mega_53_stems")
    assert len(family) == 54
    assert all(entry.sort_key for entry in family)
    ordered = sorted(family, key=lambda entry: (entry.sort_key, entry.name))
    assert ordered[0].name == "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt"
    assert [entry.name for entry in ordered[1:]] == sorted(entry.name for entry in family if entry != ordered[0])


def test_satb_labels_keep_existing_ids_and_targets():
    for name in ("model_scnet_ep_36_sdr_5.4596.ckpt", "model_scnet_ep_42_sdr_5.2559.ckpt"):
        entry = get_model_entry(name)
        assert entry.category_path == "vocal/choir_vocal"
        assert entry.secondary_category_cn == "四部和声分离"
        assert set(entry.target_stem.split("/")) == {"alto", "bass", "soprano", "tenor"}


def test_drum_extraction_classification_keeps_existing_download_paths():
    entry = get_model_entry("gilliaan_drumsV1.ckpt")
    assert entry.category_path == "single_instrument/drums_percussion"
    assert entry in list_models(category="single_instrument/drums_percussion")
    assert entry not in list_models(category="music_stems/drum_stems")
    assert entry.relpath == "music_stems/drum_stems/gilliaan_drumsV1.ckpt"
    assert entry.config_relpath == "music_stems/drum_stems/gilliaan_drumsV1.yaml"
    assert entry.target_stem == "drums/other"
