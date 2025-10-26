# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
from compel import Compel, ReturnedEmbeddingsType

_COMPEL_CACHE: dict[Tuple[int, ...], Tuple[Compel, torch.Tensor]] = {}


def _patch_compel_padding(compel_obj: Compel) -> None:
    """
    Mitigate a Compel issue where pooled embeddings can push token sequences
    past the CLIP text encoder limit by re-appending EOS after padding.
    Trim any overflow to the nearest multiple of the model max length so that
    long prompts are chunked instead of truncated.
    """
    conditioning_provider = getattr(compel_obj, "conditioning_provider", None)
    if conditioning_provider is None:
        return
    providers = getattr(conditioning_provider, "embedding_providers", None)
    if providers is None:
        providers = [conditioning_provider]

    for provider in providers:
        max_len = getattr(provider, "max_token_count", None)
        get_token_ids = getattr(provider, "get_token_ids", None)
        if not max_len or not callable(get_token_ids):
            continue

        original_get_token_ids = get_token_ids

        def patched_get_token_ids(
            self,
            texts,
            include_start_and_end_markers: bool = True,
            padding: str = "do_not_pad",
            truncation_override=None,
            _orig=original_get_token_ids,
            _max_len=max_len,
        ):
            tokens = _orig(texts, include_start_and_end_markers, padding, truncation_override)
            if not isinstance(tokens, list):
                return tokens
            trimmed = []
            for token_list in tokens:
                if not isinstance(token_list, list):
                    trimmed.append(token_list)
                    continue
                if len(token_list) > _max_len:
                    remainder = len(token_list) % _max_len
                    if remainder:
                        token_list = token_list[: len(token_list) - remainder]
                trimmed.append(token_list)
            return trimmed

        provider.get_token_ids = patched_get_token_ids.__get__(provider, provider.__class__)


def get_compel_for_sdxl(
    tokenizers: Sequence,
    text_encoders: Sequence,
    device: Union[torch.device, str, None] = None,
) -> Tuple[Compel, torch.Tensor]:
    """
    Build (or reuse) a Compel instance configured for SDXL dual-encoder setups.
    Returns a tuple of (compel_instance, empty_conditioning_tensor).
    """
    if len(tokenizers) != len(text_encoders):
        raise ValueError("tokenizers and text_encoders must be the same length")

    if any(obj is None for obj in tokenizers) or any(obj is None for obj in text_encoders):
        raise ValueError("tokenizers and text_encoders must not contain None")

    key = tuple(id(obj) for obj in (*tokenizers, *text_encoders))
    compel_obj: Compel
    empty_conditioning: torch.Tensor

    if key in _COMPEL_CACHE:
        compel_obj, cached_empty = _COMPEL_CACHE[key]
        empty_conditioning = cached_empty
    else:
        first_encoder = text_encoders[0]
        compel_device = first_encoder.device if device is None else device
        compel_obj = Compel(
            tokenizer=list(tokenizers),
            text_encoder=list(text_encoders),
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
            device=compel_device,
        )
        _patch_compel_padding(compel_obj)

        providers = getattr(compel_obj.conditioning_provider, "embedding_providers", None)
        dtype = getattr(first_encoder, "dtype", torch.float32)
        if providers:
            empty_parts = []
            for provider in providers:
                if hasattr(provider, "empty_z"):
                    part = provider.empty_z
                else:
                    part = provider.get_embeddings_for_weighted_prompt_fragments(
                        [[""]],
                        [[1.0]],
                        should_return_tokens=False,
                        device=compel_device,
                    )
                empty_parts.append(part.to(device=compel_device, dtype=dtype))
            empty_conditioning = torch.cat(empty_parts, dim=-1)
        else:
            provider = compel_obj.conditioning_provider
            empty_source = getattr(provider, "empty_z", None)
            if empty_source is None:
                empty_source = provider.get_embeddings_for_weighted_prompt_fragments(
                    [[""]],
                    [[1.0]],
                    should_return_tokens=False,
                    device=compel_device,
                )
            empty_conditioning = empty_source.to(device=compel_device, dtype=dtype)

        _COMPEL_CACHE[key] = (compel_obj, empty_conditioning)

    target_device = text_encoders[0].device if device is None else device
    target_dtype = getattr(text_encoders[0], "dtype", torch.float32)
    empty_conditioning = empty_conditioning.to(device=target_device, dtype=target_dtype)

    conditioning_provider = compel_obj.conditioning_provider
    try:
        conditioning_provider.empty_z = empty_conditioning
    except AttributeError:
        if hasattr(conditioning_provider, "__dict__"):
            conditioning_provider.__dict__["empty_z"] = empty_conditioning
        else:
            object.__setattr__(conditioning_provider, "empty_z", empty_conditioning)

    return compel_obj, empty_conditioning


__all__ = ["get_compel_for_sdxl"]
