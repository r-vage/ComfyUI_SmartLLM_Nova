# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Adapted for SmartLLM Nova native text generation.
import logging
import warnings

import torch

import gguf

from .dequant import dequantize_tensor, is_quantized
from .mappings import (
    CLIP_VISION_QWEN3_MAP,
    CLIP_VISION_SD_MAP,
    GEMMA3_SD_MAP,
    LLAMA_SD_MAP,
    sd_map_replace,
)
from .ops import GGMLTensor

logger = logging.getLogger(__name__)


def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(
            f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}",
        )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))



def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    if field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(
                f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}",
            )
        return str(field.parts[field.data[-1]], encoding="utf-8")
    if field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    raise TypeError(f"Unknown field type {field_type}")



def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    if field_type == str:
        return tuple(
            str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data
        )
    if field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    raise TypeError(f"Unknown field type {field_type}")



def gguf_sd_loader(path, handle_prefix=None):
    """Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = {tensor.name for tensor in reader.tensors}
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    if arch_str not in {"qwen2vl", "qwen3", "qwen3vl", "gemma3"} and type_str not in {"mmproj", "clip-vision"}:
        raise ValueError(f"Unsupported native GGUF text encoder architecture: {arch_str!r}")

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        # torch_tensor = torch.from_numpy(tensor.data) # mmap

        # NOTE: line above replaced with this block to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable",
            )
            torch_tensor = torch.from_numpy(tensor.data)  # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        # add to state dict
        if tensor.tensor_type in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        }:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(
            torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape,
        )

        # 1D tensors shouldn't be quantized, this is a fix for BF16
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(
                state_dict[sd_key], dtype=torch.float32,
            )

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # print loaded tensor type counts
    logger.info(
        "gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()),
    )

    # mark largest tensor for vram estimation
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {"arch_str": arch_str}
    return (state_dict, extra)



def gemma3_norm_corrections(sd):
    # Reverse change from Gemma3Model modify_tensors in llama.cpp convert script
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.norm.weight",
    ]
    corrected = 0
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
            corrected += 1
    # logger.info(f"Gemma3: Applied -1 norm correction to {corrected} tensors")
    return sd



def gguf_gemma3_tokenizer_loader(path):
    # TODO: merge into gguf_tokenizer_loader
    logger.info(
        "Attempting to recreate sentencepiece tokenizer from GGUF file metadata...",
    )
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError(
            "Please install sentencepiece and protobuf.\npip install sentencepiece protobuf",
        ) from None
    spm = model.ModelProto()
    reader = gguf.GGUFReader(path)

    spm.normalizer_spec.name = "identity"
    spm.normalizer_spec.add_dummy_prefix = False
    spm.trainer_spec.model_type = 2
    spm.trainer_spec.input_format = "tsv"
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.max_sentence_length = 4192
    spm.trainer_spec.bos_piece = "<bos>"

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktype = get_list_field(reader, "tokenizer.ggml.token_type", int)

    if not tokens or not scores or not toktype:
        raise ValueError("Missing tokenizer metadata")

    for idx in range(len(tokens)):
        piece = spm.SentencePiece()
        piece.piece = tokens[idx]
        if idx == 3:  # UNK position
            piece.type = 2  # UNK Token
            piece.score = 0.0  # UNK Score
        else:
            piece.type = toktype[idx]
            piece.score = scores[idx]
        spm.pieces.append(piece)

    spm.trainer_spec.vocab_size = len(spm.pieces)
    logger.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")

    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))



def gguf_mmproj_loader(path, arch):
    # The caller resolves and validates the exact companion; never select by directory order.
    vsd, _ = gguf_sd_loader(path)
    if arch == "gemma3":
        return gemma3_mmproj(vsd)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    if arch == "qwen3vl":
        return sd_map_replace(vsd, CLIP_VISION_QWEN3_MAP)

    # run main replacement
    vsd = sd_map_replace(vsd, CLIP_VISION_SD_MAP)

    # handle split Q/K/V
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        # filter out attentions + group
        for k, v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16),
                )

        # recombine
        for k, v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat(
                [
                    v[f"q.{suffix}"],
                    v[f"k.{suffix}"],
                    v[f"v.{suffix}"],
                ],
                dim=0,
            )
        vsd = {key: value for key, value in vsd.items()
               if not any(part in key for part in (".attn_q.", ".attn_k.", ".attn_v."))}
        del attns

    return vsd



def gemma3_mmproj(vsd):
    mapping = {
        "mm.input_projection.weight": "multi_modal_projector.mm_input_projection_weight",
        "mm.soft_emb_norm.weight": "multi_modal_projector.mm_soft_emb_norm.weight",
        "v.patch_embd.": "vision_model.embeddings.patch_embedding.",
        "v.position_embd.": "vision_model.embeddings.position_embedding.",
        "v.blk.": "vision_model.encoder.layers.",
        "v.post_ln.": "vision_model.post_layernorm.",
        ".attn_q.": ".self_attn.q_proj.", ".attn_k.": ".self_attn.k_proj.",
        ".attn_v.": ".self_attn.v_proj.", ".attn_out.": ".self_attn.out_proj.",
        ".ffn_up.": ".mlp.fc1.", ".ffn_down.": ".mlp.fc2.",
        ".ln1.": ".layer_norm1.", ".ln2.": ".layer_norm2.",
    }
    vsd = sd_map_replace(vsd, mapping)
    # These two parameters bypass custom linear operations in ComfyUI.
    projection = "multi_modal_projector.mm_input_projection_weight"
    norm = "multi_modal_projector.mm_soft_emb_norm.weight"
    vsd[projection] = dequantize_tensor(vsd[projection], dtype=torch.float32)
    vsd[norm] = dequantize_tensor(vsd[norm], dtype=torch.float32) - 1.0
    return vsd


def gguf_clip_loader(path, *, projector_path=None):
    sd, extra = gguf_sd_loader(path)
    arch = extra["arch_str"]
    temb_key = "token_embd.weight"
    if arch == "gemma3":
        sd["spiece_model"] = gguf_gemma3_tokenizer_loader(path)
    # Native token lookup/tied output heads need a regular embedding tensor.
    # Keep the transformer layers quantized, as in ComfyUI-GGUF's text loader.
    sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
    if arch == "qwen2vl":
        # BaseGenerate projects logits directly with F.linear, bypassing GGMLOps.
        sd["output.weight"] = dequantize_tensor(sd["output.weight"], dtype=torch.float16)
    sd = sd_map_replace(sd, GEMMA3_SD_MAP if arch == "gemma3" else LLAMA_SD_MAP)
    if arch == "gemma3":
        sd = gemma3_norm_corrections(sd)
    if projector_path is not None:
        sd.update(gguf_mmproj_loader(projector_path, arch))
    elif arch == "qwen3vl":
        # ComfyUI detects Qwen3-VL from these keys. They are detector sentinels,
        # never evidence of vision support; discovery marks this entry text-only.
        hidden = sd["model.embed_tokens.weight"].shape[1]
        merge_dim = 4096 if hidden == 2560 else 4608
        sd["model.visual.deepstack_merger_list.0.norm.weight"] = torch.zeros(merge_dim)
        sd["model.visual.merger.linear_fc2.weight"] = torch.zeros(hidden, merge_dim)
    return sd
