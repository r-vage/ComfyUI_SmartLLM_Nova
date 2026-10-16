import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .json_store import (
    JsonStoreError,
    read_json_object,
    update_json_object,
    write_json_object,
)
from .logger import log
from .theme_defaults import extract_theme_defaults

_LOG_PREFIX = "Defaults"
_MANIFEST_NAME = ".manifest.json"
_DEFAULT_VALUE_UPGRADES = {
    Path("config/system_prompts.json"): {
        "MiniMax Music 3": (
            "6db71d6f7fd9890a6f75127e1e290cbe72988d5b6d39d99812ee4a3cd738e739",
            "aba0293889df90a05a4a2f14c083ec10d22b75f37b3f2fa3a173dc0bd2253701",
        ),
        "MiniMax H3 Scene 5s": (
            "e03e44bae85d73581a1908c7dc21e0c898a008019096482631d743b09794ffe8",
            "0b7efa02451ded08444004329cfc3defa0b5fd9b558dd5c5493e068643d0caaa",
            "3b64d6cbbbde03aec3da19823c6d7ac229f079ccc7fc1c4f7d33012eef57b4f1",
        ),
        "MiniMax H3 T2VA Timeline 15s": (
            "4088c51d1f3c1aa8ca8d55ff67302e4b1ddd57abdd545961f64619d640d09a16",
        ),
        "MiniMax H3 I2VA Timeline 15s": (
            "a0db383415417dc60a310a0cf6bc7414af6502b8aa278bd63ff71c7e7685fe56",
        ),
        "MiniMax H3 FL2VA Timeline 15s": (
            "2bcd528f236f2ef64ee26a75cfeff64080a9e891b2ba5333b32e7a652f784a62",
        ),
        "MiniMax H3 L2VA Timeline 15s": (
            "118548380acf93fe8e2352246538b57ce1560ce4a835ca9b218dd33433899ce2",
        ),
        "Wan 2.2 Scene 5s": "4e85319ae508d07c8df224dabb11c02cd686f88ef11557071d1d7c7aae48ff5f",
        "Wan 2.2 Timeline 5s": "955c2d2efd52d093fa57ff646652615a145b0e7b46d60f95909af2c52987d5e0",
        "Wan 2.2 Timeline 5s 2s": "88d1e095216abcd1acd1721f9b63f4e104f93711e261f5b7553b6f9b15d2c0d0",
        "Wan 2.2 Timeline 5s 3s": "46761f8c47641b6d776a8b1d3ba29806f26aa48ad904b9fa3c10b2076b847225",
        "Wan 2.2 Scene 10s": "ecd4a75b5b62b9028fda5e4f23a23bad082620f61e99266f67d270fd62d59513",
        "Wan 2.2 Timeline 10s": "361d1dcd651abaf2cbbdf07224d01e96d2c3e06704dbc928bd492e895e1ab722",
        "Wan 2.2 Scene 20s": "7a919e2e6e6b286818a2cc640bac7fb580f66dd09a2628b9b9f95dd9f4ef05e2",
        "Wan 2.2 Timeline 20s": "706468776084156d1c42bbbfcbb7556b47309740dafb1561ed3b133d47a40b02",
        "Wan 2.2 CN Atomic": "7a5523da49285d67ce93f538a6b89c538ee4ab8e65900805e4b28a11321220cd",
        "LTX 2.3 I2V": "dbf829e66bc2232fb5dd542c069f804b81641e2add4cc3233c9d8def4b705eeb",
    },
    Path("config/llm_few_shot_training.json"): {
        "minimax_music_3": (
            "eb4dd0d3053de1b777d09b24fea6ef467976fbe0a3c0c5215e94705c4364cc16",
            "de9b9b0554e455f994fbc2160d9b4b87f1f21764d19dbf65c18985449ebd7919",
        ),
        "minimax_h3_scene_5s": (
            "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
            "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
            "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
        ),
        "minimax_h3_i2va_timeline_15s": (
            "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
        ),
        "minimax_h3_fl2va_timeline_15s": (
            "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
        ),
        "minimax_h3_l2va_timeline_15s": (
            "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
        ),
        "wan_2.2_scene_5s": "bd734ddd2927506eea6208fc2eaf9f3aac38601853fac3e360c439b43a6b4712",
        "wan_2.2_timeline_5s": "74ce9c85afcb14c01dcc48ce682c8cd844aabd7cedb14c3e4c5644383df2579c",
        "wan_2.2_timeline_5s_2s": "8c63941e5b244b21a6b5473afea1e2a7c438a6fbf6114f488df4fd36136166a8",
        "wan_2.2_timeline_5s_3s": "dbfc717968a30a7edfe0cf1461708a528bc30e295acf1bdf8e0be757266faf09",
        "wan_2.2_scene_10s": "289bd17589b10f0eb2bcab9b8a9e42fd37bc2d30c1884ef5ed4aa08f43b707ea",
        "wan_2.2_timeline_10s": "839803e40b31b23d11f6e234a8788a3893d87ff8a2620c7b05110a182d178714",
        "wan_2.2_scene_20s": "cfde6cefcb4e1e1af9c06004dfe5d877e75b41dde6bf8f63d395bd5db666d8c5",
        "wan_2.2_timeline_20s": "3cf9e9399d9f94bbcfe34cb8d85ddc12dc0ea931c98f5d75a1f6ae918713e53b",
        "ltx_2.3_i2v": "ddddcb90ebc4292ae937be87f8fe1d542e6ab55416fc777a56dd44208ea4a397",
    },
    Path("config/llm_few_shot_training_nsfw.json"): {
        "minimax_music_3": (
            "eb4dd0d3053de1b777d09b24fea6ef467976fbe0a3c0c5215e94705c4364cc16",
            "de9b9b0554e455f994fbc2160d9b4b87f1f21764d19dbf65c18985449ebd7919",
        ),
        "minimax_h3_scene_5s": (
            "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
            "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
            "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
        ),
        "minimax_h3_i2va_timeline_15s": (
            "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
        ),
        "minimax_h3_fl2va_timeline_15s": (
            "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
        ),
        "minimax_h3_l2va_timeline_15s": (
            "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
        ),
        "wan_2.2_scene_5s": "1c61cde6ddc8c83ef910f2bee4fa6ce644cecd6f300a76267ba000e75e793f44",
        "wan_2.2_timeline_5s": "ce2153d353b6849013343f621fd7beb47a77a0ca32eff391eb5a97905a1b10b9",
        "wan_2.2_timeline_5s_2s": "8c63941e5b244b21a6b5473afea1e2a7c438a6fbf6114f488df4fd36136166a8",
        "wan_2.2_timeline_5s_3s": "dbfc717968a30a7edfe0cf1461708a528bc30e295acf1bdf8e0be757266faf09",
        "wan_2.2_scene_10s": "1fdf84cd819dc66ca8f23fd7fc2c5e4dd452b2410e60513632888792abe5ef3f",
        "wan_2.2_timeline_10s": "95989ed6d728aede0e8e811eb4e420e2f523f6a35e79a2155dbd87323bc4dad4",
        "wan_2.2_scene_20s": "cfde6cefcb4e1e1af9c06004dfe5d877e75b41dde6bf8f63d395bd5db666d8c5",
        "wan_2.2_timeline_20s": "3cf9e9399d9f94bbcfe34cb8d85ddc12dc0ea931c98f5d75a1f6ae918713e53b",
        "ltx_2.3_i2v": "54d0a9b12e92b6eace281b066c6758d8ddc89fdb54479a215d5fa075c9e4dbbb",
    },
}

_DEFAULT_KEY_RENAMES = {
    Path("config/system_prompts.json"): {
        "MiniMax H3 Scene": (
            "MiniMax H3 Scene 5s",
            (
                "e03e44bae85d73581a1908c7dc21e0c898a008019096482631d743b09794ffe8",
                "0b7efa02451ded08444004329cfc3defa0b5fd9b558dd5c5493e068643d0caaa",
                "3b64d6cbbbde03aec3da19823c6d7ac229f079ccc7fc1c4f7d33012eef57b4f1",
                "b183658396a11a1fd5b945dec4ee5593b2466ba63cfc5664813bf6d319ef5553",
            ),
        ),
        "MiniMax H3 T2VA Timeline": (
            "MiniMax H3 T2VA Timeline 15s",
            (
                "4088c51d1f3c1aa8ca8d55ff67302e4b1ddd57abdd545961f64619d640d09a16",
                "e73f1481789e886857611e630339c6257d91fd047ffa15ea81c7a3fb023d0f23",
            ),
        ),
        "MiniMax H3 I2VA Timeline": (
            "MiniMax H3 I2VA Timeline 15s",
            (
                "a0db383415417dc60a310a0cf6bc7414af6502b8aa278bd63ff71c7e7685fe56",
                "4777b732afb13c79fd7c31894dfd16cd3676f9d3b40a9da1dd63888cbe7307b0",
            ),
        ),
        "MiniMax H3 FL2VA Timeline": (
            "MiniMax H3 FL2VA Timeline 15s",
            (
                "2bcd528f236f2ef64ee26a75cfeff64080a9e891b2ba5333b32e7a652f784a62",
                "8c953cba33cdce074ba52bd8a2a9d76b10fd40736309726d424ad1acd890374e",
            ),
        ),
        "MiniMax H3 L2VA Timeline": (
            "MiniMax H3 L2VA Timeline 15s",
            (
                "118548380acf93fe8e2352246538b57ce1560ce4a835ca9b218dd33433899ce2",
                "46635e63f79cbb235782b7def702f39c978d9d627ee348f5e22b81f89b744880",
            ),
        ),
    },
    Path("config/llm_few_shot_training.json"): {
        "minimax_h3_scene": (
            "minimax_h3_scene_5s",
            (
                "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
                "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
                "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
                "7096773b3774a2b94aa33d22251f7e29b147e7080a4f95806150f87f22c5ef2c",
            ),
        ),
        "minimax_h3_t2va_timeline": (
            "minimax_h3_t2va_timeline_15s",
            ("41216e089117981b5ad06827042880f29c6c1aa7129f43ce9c8625871ca20ca6",),
        ),
        "minimax_h3_i2va_timeline": (
            "minimax_h3_i2va_timeline_15s",
            (
                "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
                "89317f62af3daa87a8b239ac111d00d4296568543c776abc4a79c84c6454ec64",
            ),
        ),
        "minimax_h3_fl2va_timeline": (
            "minimax_h3_fl2va_timeline_15s",
            (
                "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
                "ff00012d89f2593f378c54516d19cde408c1a9037d7b7d281dae7ff4243601df",
            ),
        ),
        "minimax_h3_l2va_timeline": (
            "minimax_h3_l2va_timeline_15s",
            (
                "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
                "634a3c7ffd56365a31852a4241bcba52bc541b34d8135374280fa11af01a4998",
            ),
        ),
    },
    Path("config/llm_few_shot_training_nsfw.json"): {
        "minimax_h3_scene": (
            "minimax_h3_scene_5s",
            (
                "eecd52bf4a3e38146d684c9bbfd400074a8819eacc27e40a5b11901153bb7885",
                "a44c6a7d90bd547f0fa504e933ceadd81d3aa5db92d029ce415e637d102fc70e",
                "7d00fd34e06d265888849950e7e6e54b1cacba78c4faebe56c02614e6fe8c121",
                "7096773b3774a2b94aa33d22251f7e29b147e7080a4f95806150f87f22c5ef2c",
            ),
        ),
        "minimax_h3_t2va_timeline": (
            "minimax_h3_t2va_timeline_15s",
            ("41216e089117981b5ad06827042880f29c6c1aa7129f43ce9c8625871ca20ca6",),
        ),
        "minimax_h3_i2va_timeline": (
            "minimax_h3_i2va_timeline_15s",
            (
                "b7cf284d40b783c691c093487cc4fc1bd0902122c0d027370e51b88a18ae6aef",
                "89317f62af3daa87a8b239ac111d00d4296568543c776abc4a79c84c6454ec64",
            ),
        ),
        "minimax_h3_fl2va_timeline": (
            "minimax_h3_fl2va_timeline_15s",
            (
                "2d793ea7f37a7c632ef8fe73e0ac6eac6759a9ec44e920be6709944a21411851",
                "ff00012d89f2593f378c54516d19cde408c1a9037d7b7d281dae7ff4243601df",
            ),
        ),
        "minimax_h3_l2va_timeline": (
            "minimax_h3_l2va_timeline_15s",
            (
                "db70156ce675c9f0c5b91a8ae275275b761b726321e5950f756deccef3887940",
                "634a3c7ffd56365a31852a4241bcba52bc541b34d8135374280fa11af01a4998",
            ),
        ),
    },
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json_object(path)
    except (JsonStoreError, OSError) as error:
        log.warning(_LOG_PREFIX, f"Skipped unreadable defaults manifest '{path}': {error}")
        return None


def _read_bundled_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        log.warning(_LOG_PREFIX, f"Skipped unreadable bundled JSON '{path}': {error}")
        return None
    if not isinstance(data, dict):
        log.warning(_LOG_PREFIX, f"Skipped non-object bundled JSON '{path}'")
        return None
    return data


def _merge_missing(target: dict[str, Any], additions: dict[str, Any]) -> None:
    for key, value in additions.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_missing(target[key], value)


def _merge_bundled_update(
    relative: Path,
    current: dict[str, Any],
    bundled: dict[str, Any],
) -> None:
    # New defaults remain additive. Prompt entries are upgraded individually
    # only while they still match a recognized prior bundled value; local edits
    # and retired keys remain untouched.
    renames = _DEFAULT_KEY_RENAMES.get(relative, {})
    for new_key, (old_key, legacy_hashes) in renames.items():
        if new_key in current or new_key not in bundled or old_key not in current:
            continue
        if _value_hash(current[old_key]) in legacy_hashes:
            current[new_key] = copy.deepcopy(bundled[new_key])
        else:
            current[new_key] = copy.deepcopy(current[old_key])

    upgrades = _DEFAULT_VALUE_UPGRADES.get(relative, {})
    for key, legacy_hashes in upgrades.items():
        recognized_hashes = (
            (legacy_hashes,) if isinstance(legacy_hashes, str) else legacy_hashes
        )
        if (
            key in current
            and key in bundled
            and _value_hash(current[key]) in recognized_hashes
        ):
            current[key] = copy.deepcopy(bundled[key])
    _merge_missing(current, bundled)


def _load_manifest(defaults: Path) -> dict[str, Any]:
    manifest = _read_optional_object(defaults / _MANIFEST_NAME)
    return manifest if manifest is not None else {}


def materialize_defaults(repo_root: Path) -> None:
    extract_theme_defaults(repo_root)
    defaults = repo_root / ".defaults"
    manifest_path = defaults / _MANIFEST_NAME
    manifest = _load_manifest(defaults)
    changed = False
    extracted = 0
    updated = 0

    for example in sorted(defaults.rglob("*.example")):
        relative = example.relative_to(defaults)
        relative = Path(str(relative)[: -len(".example")])
        key = relative.as_posix()
        target = repo_root / relative
        example_hash = _file_hash(example)
        previous_hash = manifest.get(key)

        if relative.parts[0] == "themes":
            # Theme updates never modify an extracted user file or restore one
            # the user deleted. Keep only the shipped hash in this manifest.
            if previous_hash != example_hash:
                manifest[key] = example_hash
                changed = True
            continue

        if not target.exists():
            if previous_hash is not None:
                continue
            data = _read_bundled_object(example)
            if data is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            write_json_object(target, data, private=relative == Path("config.json"))
            manifest[key] = example_hash
            changed = True
            extracted += 1
        elif previous_hash is None:
            data = _read_bundled_object(example)
            if data is not None:
                update_json_object(
                    target,
                    lambda current, bundled_data=data, rel=relative: (
                        _merge_bundled_update(rel, current, bundled_data)
                    ),
                    private=relative == Path("config.json"),
                )
            manifest[key] = example_hash
            changed = True
        elif previous_hash != example_hash:
            if _file_hash(target) == previous_hash:
                data = _read_bundled_object(example)
                if data is not None:
                    # A saved preference still belongs to the user when the
                    # complete config happens to match the previous defaults.
                    if relative == Path("config.json") or relative in _DEFAULT_VALUE_UPGRADES:
                        update_json_object(
                            target,
                            lambda current, bundled_data=data, rel=relative: (
                                _merge_bundled_update(rel, current, bundled_data)
                            ),
                            private=relative == Path("config.json"),
                        )
                    else:
                        write_json_object(
                            target,
                            data,
                            private=relative == Path("config.json"),
                        )
                    updated += 1
            else:
                data = _read_bundled_object(example)
                if data is not None:
                    update_json_object(
                        target,
                        lambda current, bundled_data=data, rel=relative: (
                            _merge_bundled_update(rel, current, bundled_data)
                        ),
                        private=relative == Path("config.json"),
                    )
            manifest[key] = example_hash
            changed = True

        elif relative in _DEFAULT_VALUE_UPGRADES:
            # A manifest records the bundled file, not the presence of each task
            # in runtime data. Repair missing task defaults even when hashes match.
            # The atomic updater leaves customized values and unchanged files alone.
            data = _read_bundled_object(example)
            if data is not None:
                update_json_object(
                    target,
                    lambda current, bundled_data=data, rel=relative: (
                        _merge_bundled_update(rel, current, bundled_data)
                    ),
                    private=False,
                )

    if changed:
        write_json_object(manifest_path, manifest, private=True)
    if extracted:
        log.msg(_LOG_PREFIX, f"Extracted {extracted} bundled default file(s)")
    if updated:
        log.msg(_LOG_PREFIX, f"Updated {updated} unmodified default file(s)")
