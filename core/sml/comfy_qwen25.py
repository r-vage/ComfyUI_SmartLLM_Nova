"""Per-instance Qwen2.5 generation support for ComfyUI conditioning encoders."""


def generation_options(architecture: str, options: dict) -> dict:
    if architecture != "QWEN25_7B":
        return options
    import comfy.text_encoders.qwen_image as qwen
    import comfy.text_encoders.qwen_vl as vision

    class Qwen25Generator(qwen.Qwen25_7BVLIModel):
        def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p,
                     min_p, repetition_penalty, seed, presence_penalty=0.0, mtp=True):
            if isinstance(tokens, dict):
                tokens = next(iter(tokens.values()))
            tokens_only = [[token[0] for token in batch] for batch in tokens]
            embeds, _, _, info = self.process_tokens(tokens_only, device=self.execution_device)
            kwargs = {"stop_tokens": [151643, 151645], "presence_penalty": presence_penalty}
            if any(item.get("type") == "image" for item in info):
                positions = getattr(vision, "qwen2vl_mrope_position_ids", None)
                if not callable(positions):
                    raise RuntimeError("Update ComfyUI for native Qwen2.5-VL image generation support.")
                kwargs["position_ids"] = positions(info, embeds.shape[1], embeds.device)
            return self.transformer.generate(
                embeds, do_sample, max_length, temperature, top_k, top_p,
                min_p, repetition_penalty, seed, **kwargs,
            )

    # 7B uses an independent LM head. The conditioning class defaults to tied
    # embeddings, so request its existing model-config override before loading.
    return {**options, "qwen25_7b_class": Qwen25Generator,
            "qwen25_7b_model_config": {"lm_head": True}}
