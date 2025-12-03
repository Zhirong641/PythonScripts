# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence, Tuple, Union

import torch
from compel import Compel, ReturnedEmbeddingsType

_COMPEL_CACHE: dict[Tuple[int, ...], "_SDXLCompelWrapper"] = {}


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
            if truncation_override:
                truncated = []
                for token_list in tokens:
                    if isinstance(token_list, list) and len(token_list) > _max_len:
                        truncated.append(token_list[:_max_len])
                    else:
                        truncated.append(token_list)
                return truncated
            trimmed = []
            for token_list in tokens:
                if not isinstance(token_list, list):
                    trimmed.append(token_list)
                    continue
                if len(token_list) > _max_len:
                    remainder = len(token_list) % _max_len
                    if remainder:
                        padding_needed = _max_len - remainder
                        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                        if pad_token_id is None:
                            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
                        if pad_token_id is None:
                            pad_token_id = getattr(self.tokenizer, "bos_token_id", 0)
                        token_list = token_list + [pad_token_id] * padding_needed
                trimmed.append(token_list)
            return trimmed

        provider.get_token_ids = patched_get_token_ids.__get__(provider, provider.__class__)


class _SDXLCompelWrapper:
    """
    Minimal wrapper to keep the existing interface while avoiding the deprecated
    multi-tokenizer/text-encoder Compel constructor. Internally maintains two
    Compel instances and concatenates their outputs.
    """

    def __init__(
        self,
        compel_one: Compel,
        compel_two: Compel,
        empty_conditioning: torch.Tensor,
        empty_one: torch.Tensor,
        empty_two: torch.Tensor,
    ):
        self.compel_one = compel_one
        self.compel_two = compel_two
        self.conditioning_provider = SimpleNamespace(empty_z=empty_conditioning)
        self._empty_one = empty_one
        self._empty_two = empty_two

    @staticmethod
    def _pad_embeddings(tensor: torch.Tensor, target_tokens: int, pad_source: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] == target_tokens:
            return tensor
        pad_len = target_tokens - tensor.shape[1]
        base = pad_source.to(device=tensor.device, dtype=tensor.dtype)
        if base.dim() == 2:
            base = base.unsqueeze(0)
        base_tokens = base.shape[1]
        if base_tokens == 0:
            zeros = torch.zeros(
                (tensor.shape[0], pad_len, tensor.shape[2]),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return torch.cat([tensor, zeros], dim=1)
        repeats = math.ceil(pad_len / base_tokens)
        base_expanded = base.expand(tensor.shape[0], base_tokens, base.shape[2])
        pad_chunk = base_expanded.repeat(1, repeats, 1)[:, :pad_len, :]
        return torch.cat([tensor, pad_chunk], dim=1)

    def _concat_with_padding(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        max_tokens = max(left.shape[1], right.shape[1])
        if max_tokens == 0:
            return torch.cat([left, right], dim=-1)
        if left.shape[1] != max_tokens:
            left = self._pad_embeddings(left, max_tokens, self._empty_one)
        if right.shape[1] != max_tokens:
            right = self._pad_embeddings(right, max_tokens, self._empty_two)
        return torch.cat([left, right], dim=-1)

    def __call__(self, text):
        left_embeds = self.compel_one(text)
        right_embeds, pooled = self.compel_two(text)
        embeds = self._concat_with_padding(left_embeds, right_embeds)
        return embeds, pooled

    def pad_conditioning_tensors_to_same_length(
        self, conditionings, precomputed_padding: torch.Tensor | None = None
    ):
        if precomputed_padding is None:
            precomputed_padding = self.conditioning_provider.empty_z
        return Compel._pad_conditioning_tensors_to_same_length(
            conditionings, emptystring_conditioning=precomputed_padding
        )

    def update_empty_conditioning(
        self, empty_conditioning: torch.Tensor, empty_one: torch.Tensor, empty_two: torch.Tensor
    ) -> None:
        self.conditioning_provider.empty_z = empty_conditioning
        self._empty_one = empty_one
        self._empty_two = empty_two


def _build_empty_conditioning(compel_obj: Compel, text_encoder, device):
    providers = getattr(compel_obj.conditioning_provider, "embedding_providers", None)
    dtype = getattr(text_encoder, "dtype", torch.float32)
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
                    device=device,
                )
            empty_parts.append(part.to(device=device, dtype=dtype))
        empty_conditioning = torch.cat(empty_parts, dim=-1)
    else:
        provider = compel_obj.conditioning_provider
        empty_source = getattr(provider, "empty_z", None)
        if empty_source is None:
            empty_source = provider.get_embeddings_for_weighted_prompt_fragments(
                [[""]],
                [[1.0]],
                should_return_tokens=False,
                device=device,
            )
        empty_conditioning = empty_source.to(device=device, dtype=dtype)
    if empty_conditioning.dim() == 2:
        empty_conditioning = empty_conditioning.unsqueeze(0)
    try:
        compel_obj.conditioning_provider.empty_z = empty_conditioning
    except AttributeError:
        if hasattr(compel_obj.conditioning_provider, "__dict__"):
            compel_obj.conditioning_provider.__dict__["empty_z"] = empty_conditioning
        else:
            object.__setattr__(compel_obj.conditioning_provider, "empty_z", empty_conditioning)
    return empty_conditioning


def _concat_empty_conditionings(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape[1] == right.shape[1]:
        return torch.cat([left, right], dim=-1)
    max_tokens = max(left.shape[1], right.shape[1])
    if left.shape[1] != max_tokens:
        if left.shape[1] == 0:
            left = torch.zeros((left.shape[0], max_tokens, left.shape[2]), device=left.device, dtype=left.dtype)
        else:
            pad_len = max_tokens - left.shape[1]
            repeats = math.ceil(pad_len / left.shape[1])
            left = left.repeat(1, repeats + 1, 1)[:, :max_tokens, :]
    if right.shape[1] != max_tokens:
        if right.shape[1] == 0:
            right = torch.zeros((right.shape[0], max_tokens, right.shape[2]), device=right.device, dtype=right.dtype)
        else:
            pad_len = max_tokens - right.shape[1]
            repeats = math.ceil(pad_len / right.shape[1])
            right = right.repeat(1, repeats + 1, 1)[:, :max_tokens, :]
    return torch.cat([left, right], dim=-1)


def get_compel_for_sdxl(
    tokenizers: Sequence,
    text_encoders: Sequence,
    device: Union[torch.device, str, None] = None,
) -> Tuple[_SDXLCompelWrapper, torch.Tensor]:
    """
    Build (or reuse) a Compel instance configured for SDXL dual-encoder setups.
    Returns a tuple of (compel_instance, empty_conditioning_tensor).
    """
    if len(tokenizers) != len(text_encoders):
        raise ValueError("tokenizers and text_encoders must be the same length")

    if any(obj is None for obj in tokenizers) or any(obj is None for obj in text_encoders):
        raise ValueError("tokenizers and text_encoders must not contain None")

    key = tuple(id(obj) for obj in (*tokenizers, *text_encoders))

    if key in _COMPEL_CACHE:
        compel_wrapper = _COMPEL_CACHE[key]
    else:
        first_encoder = text_encoders[0]
        compel_device = first_encoder.device if device is None else device
        compel_one = Compel(
            tokenizer=tokenizers[0],
            text_encoder=text_encoders[0],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=False,
            truncate_long_prompts=False,
            device=compel_device,
        )
        compel_two = Compel(
            tokenizer=tokenizers[1],
            text_encoder=text_encoders[1],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            truncate_long_prompts=False,
            device=compel_device,
        )
        _patch_compel_padding(compel_one)
        _patch_compel_padding(compel_two)

        empty_one = _build_empty_conditioning(compel_one, text_encoders[0], compel_device)
        empty_two = _build_empty_conditioning(compel_two, text_encoders[1], compel_device)
        empty_conditioning = _concat_empty_conditionings(empty_one, empty_two)

        compel_wrapper = _SDXLCompelWrapper(compel_one, compel_two, empty_conditioning, empty_one, empty_two)
        _COMPEL_CACHE[key] = compel_wrapper

    target_device = text_encoders[0].device if device is None else device
    text_dtype = getattr(text_encoders[0], "dtype", torch.float32)
    pooled_dtype = getattr(text_encoders[-1], "dtype", text_dtype)

    empty_one = compel_wrapper._empty_one.to(device=target_device, dtype=text_dtype)
    empty_two = compel_wrapper._empty_two.to(device=target_device, dtype=pooled_dtype)
    empty_conditioning = _concat_empty_conditionings(empty_one, empty_two)
    compel_wrapper.update_empty_conditioning(empty_conditioning, empty_one, empty_two)

    return compel_wrapper, empty_conditioning


__all__ = ["get_compel_for_sdxl"]
