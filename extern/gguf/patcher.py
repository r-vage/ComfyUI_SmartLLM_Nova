# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Adapted for SmartLLM Nova: patcher only, without node/folder registration.
import collections
import logging
from contextlib import contextmanager

import comfy.float
import comfy.lora
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import torch

from .dequant import is_quantized, is_torch_compatible
from .ops import move_patch_to_device

logger = logging.getLogger(__name__)


class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    generation_memory_required = 0

    @contextmanager
    def reserve_generation_memory(self, memory_required):
        previous = self.generation_memory_required
        self.generation_memory_required = max(0, memory_required)
        try:
            yield
        finally:
            self.generation_memory_required = previous

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        mm = comfy.model_management
        reserved = 0
        before = self.loaded_size()
        if (self.generation_memory_required and not mm.is_device_cpu(device_to)
                and device_to != self.offload_device):
            reserved = max(mm.minimum_inference_memory(),
                           self.generation_memory_required + mm.extra_reserved_memory())
            available = mm.get_free_memory(device_to)
            if available + before <= reserved:
                raise RuntimeError(
                    f"Qwen GGUF generation needs approximately {reserved / 1024**3:.1f} GiB "
                    "of free workspace. Reduce the input image size, prompt length or max_tokens."
                )
            if before:
                # Do not refill a cached model into its generation workspace.
                extra_memory = min(extra_memory, available - reserved)
        result = super().partially_load(device_to, extra_memory, force_patch_weights)
        if reserved:
            # Let the native loader prioritize large weights first, then evict
            # smaller layers for generation. Reducing its initial load budget
            # double-counts native transfer buffers and can offload the entire
            # model merely because its vocabulary tables are large.
            missing = reserved - mm.get_free_memory(device_to)
            if missing > 0:
                self.partially_unload(self.offload_device, missing, force_patch_weights=force_patch_weights)
            return self.loaded_size() - before
        return result

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(
                patches,
                self.load_device if self.patch_on_device else self.offload_device,
            )
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple(
                    "Dimension", ["weight", "inplace_update"],
                )(
                    weight.to(device=self.offload_device, copy=inplace_update),
                    inplace_update,
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(
                    weight, device_to, torch.float32, copy=True,
                )
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(
            device_to=device_to, unpatch_weights=unpatch_weights,
        )

    def pin_weight_to_device(self, key):
        op_key = key.rsplit(".", 1)[0]
        if not self.mmap_released and self.named_modules_to_munmap and op_key in self.named_modules_to_munmap:
            self._detach_mmap(self.named_modules_to_munmap[op_key])
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    def _detach_mmap(self, module):
        # Copy directly on the offload device. A GPU round trip can OOM even
        # though the layer is deliberately excluded from the resident weights.
        for name, parameter in module.named_parameters(recurse=False):
            copied = parameter.to(device=self.offload_device, copy=True)
            comfy.utils.set_attr_param(module, name, copied)
        for name, buffer in module.named_buffers(recurse=False):
            setattr(module, name, buffer.to(device=self.offload_device, copy=True))

    mmap_released = False
    named_modules_to_munmap = None

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logger.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    self._detach_mmap(m)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        try:
            n = super().clone(*args, **kwargs)
        finally:
            self.__class__ = src_cls
        n.__class__ = GGUFModelPatcher
        # GGUF specific clone values below
        n.named_modules_to_munmap = dict(getattr(self, "named_modules_to_munmap", None) or {})
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        n.generation_memory_required = 0
        if src_cls != GGUFModelPatcher:
            n.size = 0  # force recalc
        return n


