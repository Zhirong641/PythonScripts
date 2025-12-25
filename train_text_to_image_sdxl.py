#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image."""

import argparse
import functools
import gc
import logging
import math
import os
import random
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
import json
from itertools import chain
from typing import Optional
from collections import defaultdict
import time

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop, resize
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import Sampler

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from compel_sdxl_utils import get_compel_for_sdxl

# plotting (offscreen)
import csv as _pycsv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prompt_variants import generate_variants_with_nl_list
from PIL import Image, ImageFile

os.environ["TOKENIZERS_PARALLELISM"] = "false"
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow partially downloaded/corrupted files instead of crashing.

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    import torch_npu

    torch.npu.config.allow_internal_format = False

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

FILTER_LIST_DIR = Path(__file__).resolve().parent / "filter_lists"

# ===== Helpers for dynamic text variants =====
def _split_clean_comma_list(s: str):
    if not s:
        return []
    return [x.strip() for x in s.replace("，", ",").split(",") if x.strip()]


def _normalize_artist_tags(artist_tags):
    rename_map = {"any": "annie",
                  "kino": "konomi",
                  "anapon": "anapom",
                  "fumi": "fummy",
                  "narumi yu": "narumi yuu",
                  "akizora momidi": "akizora momiji",
                  "moeki yuta": "moeki yuuta",
                  "shira ichigo": "shiraichigo",
                  "hinata momoko": "hinata momo",
                  "yuunagi sesina": "yuunagi seshina",}
    normalized = []
    seen = set()
    for tag in artist_tags or []:
        if not tag:
            continue
        tag_norm = rename_map.get(tag.lower(), tag)
        tag_norm = tag_norm.replace("_", " ").strip()
        key = tag_norm.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(tag_norm)
    return normalized


def _join_with_comma(items):
    return ", ".join(items)


def _build_variants_from_cap_author(caption_tags: str, caption_nl: str, author: str):
    """
    Build several prompt variants using tags / natural caption / author info.
    Returns (texts: list[str], mask: np.ndarray[bool], preview_text: str)
    """
    tags = _split_clean_comma_list(caption_tags)
    auth = _split_clean_comma_list(author)
    tags_fwd = _join_with_comma(tags) if tags else ""
    if random.random() < 0.8 and auth:
        tags_fwd = f"artist:{random.choice(auth)}, {tags_fwd}" if tags_fwd else f"artist:{random.choice(auth)}"
    tags_rev = _join_with_comma(list(reversed(tags))) if tags else ""
    if random.random() < 0.8 and auth:
        tags_rev = f"artist:{random.choice(auth)}, {tags_rev}" if tags_rev else f"artist:{random.choice(auth)}"
    if auth:
        caption_auth_nl = (
            f"{caption_nl}, artist:{random.choice(auth)}" if caption_nl else f"artist:{random.choice(auth)}"
        )
    else:
        caption_auth_nl = caption_nl
    texts = [tags_fwd, tags_rev, caption_nl, caption_auth_nl]
    mask = [bool(t) for t in texts]
    # remove duplicates
    for j in range(len(texts)):
        for k in range(j):
            if mask[j] and mask[k] and texts[j] == texts[k]:
                mask[j] = False
    mask = np.array(mask, dtype=np.bool_)
    preview_text = random.choice([texts[i] for i in range(len(texts)) if mask[i]]) if any(mask) else ""
    return texts, mask, preview_text


def _load_filter_list(filename: str):
    file_path = FILTER_LIST_DIR / filename
    if not file_path.is_file():
        logger.warning(f"Filter list file '{file_path}' not found. Continuing without exclusions.")
        return []
    entries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            entries.append(item)
    return entries


def _build_filter_name_set(entries):
    names = set()
    for name in entries:
        if not name:
            continue
        names.add(name)
        names.add(name.replace("_", " "))
    return {n for n in names if n}


def _annotate_target_size(example, height, width, label=None):
    example["target_height"] = int(height)
    example["target_width"] = int(width)
    if label is not None:
        example["resolution_tag"] = label
    return example


class ResolutionBucketBatchSampler(Sampler):
    def __init__(self, bucket_map, batch_size, shuffle=True, seed=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive for ResolutionBucketBatchSampler.")
        self.bucket_map = {k: list(v) for k, v in bucket_map.items()}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed + int(time.time())) if self.seed is not None else random
        batches = []
        for _, indices in self.bucket_map.items():
            if self.shuffle:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.bucket_map.values():
            total += len(indices) // self.batch_size
        return total


def _load_resolution_sets_config(config_path: str, default_train_dir: Optional[str]):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("`--resolution_sets` must point to a JSON list with at least one entry.")
    configs = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError("Each entry inside `--resolution_sets` must be a JSON object.")
        res_value = entry.get("resolution")
        if not res_value:
            raise ValueError("Each resolution set must define a 'resolution' field.")
        height, width = _parse_resolution(res_value)
        index_file = entry.get("index_file")
        if not index_file:
            raise ValueError("Each resolution set must define an 'index_file'.")
        index_file = os.path.abspath(os.path.expanduser(index_file))
        train_dir = entry.get("train_data_dir", default_train_dir)
        if train_dir is not None:
            train_dir = os.path.abspath(os.path.expanduser(train_dir))
        label = entry.get("label") or entry.get("name") or f"set_{idx}"
        configs.append(
            {
                "resolution_height": height,
                "resolution_width": width,
                "index_file": index_file,
                "train_data_dir": train_dir,
                "label": label,
            }
        )
    return configs


def _escape_prompt_weight_syntax(text: str) -> str:
    """
    Escape weight delimiters so Compel won't treat parentheses/brackets as weights.
    """
    if not text:
        return text
    pattern = re.compile(r"(?<!\\)([()\[\]{}])")
    return pattern.sub(lambda m: "\\" + m.group(1), text)


def _load_and_filter_index_dataset(
    index_path,
    train_data_dir,
    cache_dir,
    exclude_word_list,
    exclude_artist_set,
    exclude_danbooru_set,
    exclude_source_id_set,
    include_source_id_set=None,
    seed=None,
):
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"index.jsonl not found: {index_path}")
    dataset = load_dataset(
        "json",
        data_files={"train": index_path},
        cache_dir=cache_dir,
    )
    base_dir = train_data_dir if train_data_dir else os.path.dirname(index_path)

    def _resolve_src(example):
        src = example.get("src", "") or example.get("path", "") or ""
        if src and not os.path.isabs(src):
            src_path = os.path.join(base_dir, src) if base_dir else src
        else:
            src_path = src
        example["_image_path"] = src_path
        return example

    dataset["train"] = dataset["train"].map(_resolve_src)

    rng = random.Random(seed if seed is not None else 42)

    total_exclude_set = set()
    if exclude_artist_set:
        total_exclude_set |= exclude_artist_set
    if exclude_danbooru_set:
        total_exclude_set |= exclude_danbooru_set

    def _filter_index_entry(example):
        src_path = example.get("_image_path", "") or ""
        type_ = example.get("type", "") 
        if not src_path or not os.path.isfile(src_path):
            return False
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".heic"}
        _, ext = os.path.splitext(src_path)
        ext = ext.split("?")[0].lower()
        if ext not in valid_exts:
            return False
        norm_path = os.path.normpath(src_path)
        path_parts = Path(norm_path).parts
        if exclude_source_id_set and any(part in exclude_source_id_set for part in path_parts):
            return False

        general = example.get("general", "") or ""
        general_tags = _split_clean_comma_list(general)
        artist = example.get("artist", "") or ""
        artists = _split_clean_comma_list(artist)
        if any(word in general for word in exclude_word_list):
            return False
        if ("danbooru" not in src_path
            and any(bg in general_tags for bg in ["transparent_background", "simple_background", "black_background", "white_background", "tachi-e"])
            and "dakimakura_(medium)" not in general_tags):
            group = example.get("group", "") or ""
            if ("dmm.com" not in group and 
                not (include_source_id_set and any(part in include_source_id_set for part in path_parts)) and
                type_.lower() == "game cg" and 
                rng.random() < 0.66):
                return False
        year = example.get("year", "") or ""
        years = []
        for y in _split_clean_comma_list(year):
            if y.startswith("year_") and y[5:].isdigit():
                years.append(int(y[5:]))
        if years:
            if min(years) <= 2005:
                return False
            if max(years) <= 2007 and rng.random() < 0.9:
                return False
            if max(years) <= 2009 and rng.random() < 0.7:
                return False
        meta = example.get("meta", "") or ""
        if "lowres" in meta and "highres" not in meta:
            return False

        if exclude_artist_set and any(a in exclude_artist_set for a in artists):
            return False
        if total_exclude_set and artists and all(a in total_exclude_set for a in artists):
            return False
        if "mizunezumi" in artists and rng.random() < 0.9:
            return False
        if ("danbooru" not in src_path and "ko-cha" in artists and rng.random() < 0.9):
            return False
        return True

    dataset["train"] = dataset["train"].filter(_filter_index_entry)
    if len(dataset["train"]) == 0:
        raise RuntimeError(f"No valid samples found in {index_path}")

    dataset["train"] = dataset["train"].cast_column("_image_path", datasets.Image())
    dataset["train"] = dataset["train"].rename_column("_image_path", "image")
    return dataset


# ===== Resolution helpers =====
def _parse_resolution(res_arg):
    """Parse resolution argument into (height, width)."""
    if isinstance(res_arg, int):
        if res_arg <= 0:
            raise ValueError("Resolution value must be positive")
        return res_arg, res_arg
    if isinstance(res_arg, (tuple, list)):
        if len(res_arg) == 1:
            return _parse_resolution(int(res_arg[0]))
        if len(res_arg) == 2:
            height = int(res_arg[0])
            width = int(res_arg[1])
            if height <= 0 or width <= 0:
                raise ValueError("Resolution dimensions must be positive")
            return height, width
        raise ValueError("Resolution must have one or two values")

    res_str = str(res_arg).lower().replace("×", "x").strip()
    res_str = res_str.replace(",", "x")
    parts = [p for p in res_str.replace(" ", "x").split("x") if p]

    if len(parts) == 1:
        return _parse_resolution(int(parts[0]))
    if len(parts) == 2:
        width = int(parts[0])
        height = int(parts[1])
        if height <= 0 or width <= 0:
            raise ValueError("Resolution dimensions must be positive")
        return height, width

    raise ValueError(f"Cannot parse resolution value: {res_arg}")


# ===== Latent dataset (index.jsonl + .npz) =====
class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str):
        self.root = data_dir
        index_path = os.path.join(data_dir, "index.jsonl")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"index.jsonl not found in {data_dir}")
        self.items = []
        with open(index_path, "r", encoding="utf-8") as f:
            exclude_word_list = [
                "no_humans", "chibi", "character_profile", "lineart", "sketch",
                "monochrome", "comic", "text_focus", "1990s", "1980s",
                "retro_artstyle", "abstract"
            ]
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    print(f"[DBG] Warning: failed to parse line: {line}, skipping")
                    continue
                fname = j.get("npz")
                if not fname:
                    continue

                def filter_fn(sample):
                    _, general, _rating, _, year, character, artist, _, _group, _type = sample
                    if any(w in general for w in exclude_word_list):
                        return False
                    if year:
                        years = [int(y.split("_")[1]) for y in _split_clean_comma_list(year) if y.startswith("year_") and y[5:].isdigit()]
                        if years and min(years) < 2005:
                            return False
                    return True
                fp = os.path.join(data_dir, fname)
                if os.path.isfile(fp):
                    sample = (
                        fname,
                        j.get("general", "") or "",
                        j.get("rating", "") or "",
                        j.get("meta", "") or "",
                        j.get("year", "") or "",
                        j.get("character", "") or "",
                        j.get("artist", "") or "",
                        j.get("copyright", "") or "",
                        j.get("group", "") or "",
                        j.get("type", "") or "",
                    )
                    if filter_fn(sample):
                        self.items.append(sample)
                print(f"Loaded {len(self.items)} items from {data_dir}", end="\r")
        if len(self.items) == 0:
            raise RuntimeError(f"No valid items in {index_path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, general, rating, meta, year, character, artist, copyright, group, type_val = self.items[i]
        z = np.load(os.path.join(self.root, fname), allow_pickle=False)
        lat = z["latent"].astype(np.float16)  # already scaled
        return torch.from_numpy(lat), general, rating, meta, year, character, artist, copyright, group, type_val


def save_model_card(
    repo_id: str,
    images: list = None,
    validation_prompt: str = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: \n
{img_str}

Special VAE used for training: {vae_path}.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers-training",
        "diffusers",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=None,
        help="Path to the index file (index.jsonl) for the dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help=(
            "Use a latent dataset directory (contains index.jsonl + .npz latents) instead of raw images."
            " Example: ../latent_db"
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=str,
        default="1024",
        help=(
            "The target resolution for input images. Provide a single integer for square training or"
            " a WIDTHxHEIGHT pair (e.g. 1024x768) for non-square images."
        ),
    )
    parser.add_argument(
        "--resolution_sets",
        type=str,
        default=None,
        help=(
            "JSON file describing multiple resolution/index configurations. Each entry should contain"
            " at least 'resolution' and 'index_file', and optionally 'train_data_dir'."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--precompute_text_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to precompute all text embeddings before training. Disable to compute prompts on the fly"
            " and skip the initial `datasets.map` stage."
        ),
    )
    parser.add_argument(
        "--precompute-text-embeddings",
        dest="precompute_text_embeddings",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-precompute-text-embeddings",
        dest="precompute_text_embeddings",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--precompute_vae_latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to precompute VAE latents before training. This is only supported when text embeddings are"
            " also precomputed."
        ),
    )
    parser.add_argument(
        "--precompute-vae-latents",
        dest="precompute_vae_latents",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-precompute-vae-latents",
        dest="precompute_vae_latents",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--text_encode_batch_size",
        type=int,
        default=None,
        help=(
            "Batch size to use while precomputing text embeddings. Defaults to the training batch size; raise this"
            " to keep the GPU busier during `datasets.map`."
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=None,
        help=(
            "Batch size to use while precomputing VAE latents. Defaults to the training batch size; increase this"
            " to speed up the `datasets.map` stage if you have spare VRAM."
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://huggingface.co/papers/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Fine-tune both text encoders alongside the UNet.",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        help=(
            "Learning rate applied to text encoder parameters when `--train_text_encoder` is set."
            " Defaults to 10%% of the UNet learning rate if not provided."
        ),
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[
            f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")
        ],
        help="The image interpolation method to use for resizing images.",
    )
    # Plot/preview controls
    parser.add_argument(
        "--plot_interval",
        type=int,
        default=0,
        help="If >0, save loss.csv and loss.png every N optimization steps.",
    )
    parser.add_argument(
        "--preview_save_steps",
        type=int,
        default=0,
        help="If >0, save a preview image every N optimization steps.",
    )
    parser.add_argument(
        "--preview_steps",
        type=int,
        default=25,
        help="Preview sampling steps when saving previews.",
    )
    parser.add_argument(
        "--preview_scale",
        type=float,
        default=5.5,
        help="CFG scale when saving previews.",
    )
    parser.add_argument(
        "--preview_seed",
        type=int,
        default=None,
        help="Optional fixed seed for previews.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.resolution_sets:
        config_path = os.path.abspath(os.path.expanduser(args.resolution_sets))
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"`--resolution_sets` file not found: {config_path}")
        args.resolution_sets_config = _load_resolution_sets_config(config_path, args.train_data_dir)
        first = args.resolution_sets_config[0]
        args.resolution_height = first["resolution_height"]
        args.resolution_width = first["resolution_width"]
        args.resolution = (args.resolution_height, args.resolution_width)
    else:
        try:
            res_height, res_width = _parse_resolution(args.resolution)
        except ValueError as err:
            raise ValueError(
                "`--resolution` must be a positive integer or WIDTHxHEIGHT pair (e.g. 1024x768)."
            ) from err
        args.resolution = (res_height, res_width)
        args.resolution_height = res_height
        args.resolution_width = res_width
        args.resolution_sets_config = None

    # Sanity checks
    if (
        args.dataset_dir is None
        and args.dataset_name is None
        and args.train_data_dir is None
        and args.index_file is None
        and not args.resolution_sets
    ):
        raise ValueError(
            "Need either a latent dataset_dir, a dataset name, a training folder, or to provide `--resolution_sets`."
        )
    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if not args.precompute_text_embeddings and args.precompute_vae_latents:
        raise ValueError(
            "`--precompute_vae_latents` requires text embeddings to be precomputed as well."
            " Use `--no-precompute-vae-latents` when disabling text precomputation."
        )
    if args.train_text_encoder and args.precompute_text_embeddings:
        raise ValueError(
            "`--train_text_encoder` cannot be combined with precomputed text embeddings."
            " Remove `--precompute_text_embeddings` to allow gradient updates."
        )

    if args.precompute_text_embeddings:
        if args.text_encode_batch_size is None or args.text_encode_batch_size <= 0:
            args.text_encode_batch_size = args.train_batch_size
    else:
        args.text_encode_batch_size = None

    if args.precompute_vae_latents:
        if args.vae_encode_batch_size is None or args.vae_encode_batch_size <= 0:
            args.vae_encode_batch_size = args.train_batch_size
    else:
        args.vae_encode_batch_size = None

    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    batch,
    text_encoders,
    tokenizers,
    proportion_empty_prompts,
    caption_column,
    metadata_columns=None,
    is_train=True,
):
    prompt_batch = batch[caption_column] if caption_column is not None and caption_column in batch else None

    metadata_batch = None
    if metadata_columns is not None:
        if all(col in batch for col in metadata_columns.values()):
            metadata_batch = {key: batch[col] for key, col in metadata_columns.items()}

    base_captions = []
    if metadata_batch is not None:
        general_values = metadata_batch.get("general", [])
        if not isinstance(general_values, (list, tuple)):
            general_values = [general_values]
        num_samples = len(general_values)

        def _values_for(key):
            values = metadata_batch.get(key)
            if values is None:
                return [""] * num_samples
            if not isinstance(values, (list, tuple)):
                return [values] * num_samples
            if len(values) != num_samples:
                return list(values) + [""] * max(0, num_samples - len(values))
            return values

        rating_values = _values_for("rating")
        meta_values = _values_for("meta")
        year_values = _values_for("year")
        character_values = _values_for("character")
        artist_values = _values_for("artist")
        group_values = _values_for("group")
        type_values = _values_for("type")

        for idx in range(num_samples):
            general_tags = _split_clean_comma_list(general_values[idx] or "")
            artist_tags = _normalize_artist_tags(_split_clean_comma_list(artist_values[idx] or ""))
            rating_tags = _split_clean_comma_list(rating_values[idx] or "")
            year_tags = _split_clean_comma_list(year_values[idx] or "")
            character_tags = _split_clean_comma_list(character_values[idx] or "")
            meta_tags = _split_clean_comma_list(meta_values[idx] or "")
            nl_texts = meta_tags if meta_tags else None
            group_tags = _split_clean_comma_list(group_values[idx] or "")
            type_value = type_values[idx] if idx < len(type_values) else ""
            variants = generate_variants_with_nl_list(
                general_tags,
                artist_tags,
                k=1,
                token_budget=200,
                head_keep=random.choices([12,14,16],[0.5,0.35,0.15])[0],
                dropout=0.2,
                max_general_per_variant=50,
                characters=character_tags,
                ratings=rating_tags,
                years=year_tags,
                nl_texts=nl_texts,
                groups=group_tags,
                type=type_value,
            )
            base_captions.append(variants[0] if variants else "")

    if not base_captions and prompt_batch is not None:
        for caption in prompt_batch:
            base_captions.append(caption)

    if not base_captions:
        raise ValueError("No captions available for encoding. Provide captions or valid metadata columns.")

    captions = []
    for caption in base_captions:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            captions.append(str(caption))

    device = text_encoders[0].device
    compel_obj, _ = get_compel_for_sdxl(tokenizers, text_encoders, device)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = compel_obj(captions)

    prompt_embeds = prompt_embeds.to(device=device, dtype=text_encoders[0].dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=text_encoders[-1].dtype)

    return {
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    }


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor

    # There might have slightly performance improvement
    # by changing model_input.cpu() to accelerator.gather(model_input)
    return {"model_input": model_input.cpu()}


def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] *= args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.train_text_encoder:

        def _tokenize_long_prompt(tokenizer, text, device):
            max_len = tokenizer.model_max_length
            chunk_capacity = max_len - 2
            bos = tokenizer.bos_token_id
            eos = tokenizer.eos_token_id
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

            tokens = tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]

            token_chunks = []
            mask_chunks = []
            if not tokens:
                tokens = []
            for start in range(0, len(tokens), chunk_capacity):
                chunk_tokens = tokens[start : start + chunk_capacity]
                chunk = [bos] + chunk_tokens + [eos]
                mask = [1] * len(chunk)
                if len(chunk) < max_len:
                    pad_len = max_len - len(chunk)
                    chunk.extend([pad] * pad_len)
                    mask.extend([0] * pad_len)
                token_chunks.append(chunk)
                mask_chunks.append(mask)

            if not token_chunks:
                base = [bos, eos]
                pad_len = max_len - len(base)
                token_chunks.append(base + [pad] * pad_len)
                mask_chunks.append([1, 1] + [0] * pad_len)

            input_ids = torch.tensor(token_chunks, dtype=torch.long, device=device)
            attention_mask = torch.tensor(mask_chunks, dtype=torch.long, device=device)
            return input_ids, attention_mask

        def _forward_encoder(encoder, input_ids, attention_mask):
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-2]
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            flat_mask = attention_mask.reshape(-1).bool()
            gathered = flat_hidden[flat_mask].unsqueeze(0)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                pooled = pooled[:1]
            return gathered, pooled

        def _encode_texts_with_trainable_encoders(texts):
            prompt_tensors = []
            pooled_tensors = []
            max_tokens = 0

            for text in texts:
                ids_one, mask_one = _tokenize_long_prompt(tokenizer_one, text, text_encoder_one.device)
                ids_two, mask_two = _tokenize_long_prompt(tokenizer_two, text, text_encoder_two.device)

                hidden_one, _ = _forward_encoder(text_encoder_one, ids_one, mask_one)
                hidden_two, pooled_two = _forward_encoder(text_encoder_two, ids_two, mask_two)

                seq_len = max(hidden_one.shape[1], hidden_two.shape[1])
                if hidden_one.shape[1] != seq_len:
                    pad_len = seq_len - hidden_one.shape[1]
                    pad = hidden_one.new_zeros((1, pad_len, hidden_one.shape[2]))
                    hidden_one = torch.cat([hidden_one, pad], dim=1)
                if hidden_two.shape[1] != seq_len:
                    pad_len = seq_len - hidden_two.shape[1]
                    pad = hidden_two.new_zeros((1, pad_len, hidden_two.shape[2]))
                    hidden_two = torch.cat([hidden_two, pad], dim=1)

                combined = torch.cat([hidden_one, hidden_two], dim=-1)
                prompt_tensors.append(combined)
                if pooled_two is None:
                    pooled_two = hidden_two[:, 0, :]
                pooled_tensors.append(pooled_two)
                max_tokens = max(max_tokens, combined.shape[1])

            padded_prompts = []
            for prompt in prompt_tensors:
                if prompt.shape[1] < max_tokens:
                    pad_len = max_tokens - prompt.shape[1]
                    pad = prompt.new_zeros((prompt.shape[0], pad_len, prompt.shape[2]))
                    prompt = torch.cat([prompt, pad], dim=1)
                padded_prompts.append(prompt)

            prompt_batch = torch.cat(padded_prompts, dim=0)
            pooled_batch = torch.cat(pooled_tensors, dim=0)

            prompt_batch = prompt_batch.to(accelerator.device, dtype=weight_dtype)
            pooled_batch = pooled_batch.to(accelerator.device, dtype=weight_dtype)
            return prompt_batch, pooled_batch

    # Freeze vae and optionally text encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(args.train_text_encoder)
    text_encoder_two.requires_grad_(args.train_text_encoder)

    if args.train_text_encoder:
        text_encoder_one.train()
        text_encoder_two.train()
    else:
        text_encoder_one.eval()
        text_encoder_two.eval()

    # Set unet as trainable.
    unet.train()

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder_dtype = (
        torch.float32 if args.train_text_encoder and accelerator.mixed_precision == "fp16" else weight_dtype
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=text_encoder_dtype)
    text_encoder_two.to(accelerator.device, dtype=text_encoder_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                to_save = [("unet", accelerator.unwrap_model(unet))]
                if args.train_text_encoder:
                    to_save.extend(
                        [
                            ("text_encoder", accelerator.unwrap_model(text_encoder_one)),
                            ("text_encoder_2", accelerator.unwrap_model(text_encoder_two)),
                        ]
                    )

                for name, model_to_save in to_save:
                    save_path = os.path.join(output_dir, name)
                    model_to_save.save_pretrained(save_path)
                torch.save({}, os.path.join(output_dir, "pytorch_model.bin"))

            # pop everything so accelerate doesn't try to handle it again
            while weights:
                weights.pop()
            while models:
                models.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            load_targets = [
                ("unet", unet, UNet2DConditionModel, "unet"),
            ]
            if args.train_text_encoder:
                load_targets.extend(
                    [
                        ("text_encoder", text_encoder_one, type(accelerator.unwrap_model(text_encoder_one)), "text_encoder"),
                        ("text_encoder_2", text_encoder_two, type(accelerator.unwrap_model(text_encoder_two)), "text_encoder_2"),
                    ]
                )

            for name, model_ref, model_cls, subfolder in load_targets:
                target = accelerator.unwrap_model(model_ref)
                target_dtype = next(target.parameters()).dtype
                load_path = os.path.join(input_dir, name)
                if not os.path.isdir(load_path):
                    # Fallback to default location (for backwards compatibility)
                    load_model = model_cls.from_pretrained(input_dir, subfolder=subfolder)
                else:
                    load_model = model_cls.from_pretrained(load_path)
                if isinstance(target, UNet2DConditionModel):
                    target.register_to_config(**load_model.config)
                target.load_state_dict(load_model.state_dict())
                target.to(accelerator.device, dtype=target_dtype)
                del load_model

            while models:
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    lr_scale = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    if args.scale_lr:
        args.learning_rate = args.learning_rate * lr_scale
        if args.text_encoder_lr is not None:
            args.text_encoder_lr = args.text_encoder_lr * lr_scale

    if args.train_text_encoder and args.text_encoder_lr is None:
        args.text_encoder_lr = args.learning_rate * 0.1

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    if args.train_text_encoder:
        text_encoder_params = chain(text_encoder_one.parameters(), text_encoder_two.parameters())
        params_to_optimize = [
            {"params": unet.parameters(), "lr": args.learning_rate},
            {"params": text_encoder_params, "lr": args.text_encoder_lr},
        ]
    else:
        params_to_optimize = unet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Branch A: latent dataset directory (index.jsonl + .npz latents)
    using_latent_db = args.dataset_dir is not None

    if using_latent_db:
        latent_dataset = LatentDataset(args.dataset_dir)
        # simple collate
        def collate_latent(examples):
            latents = torch.stack([ex[0] for ex in examples])
            general = [ex[1] for ex in examples]
            rating = [ex[2] for ex in examples]
            meta = [ex[3] for ex in examples]
            year = [ex[4] for ex in examples]
            character = [ex[5] for ex in examples]
            artist = [ex[6] for ex in examples]
            copyright = [ex[7] for ex in examples]
            group = [ex[8] for ex in examples]
            type_values = [ex[9] for ex in examples]

            return {
                "model_input": latents,
                "general": general,
                "rating": rating,
                "meta": meta,
                "year": year,
                "character": character,
                "artist": artist,
                "copyright": copyright,
                "group": group,
                "type": type_values,
            }

        train_dataloader = torch.utils.data.DataLoader(
            latent_dataset,
            shuffle=True,
            collate_fn=collate_latent,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        # No precomputation, keep text encoders for runtime encoding
        precomputed_dataset = None

        # ====== Scheduler/Accelerate prepare (latent path) ======
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Prepare modules
        if args.train_text_encoder:
            (
                unet,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
            )
            text_encoder_one.train()
            text_encoder_two.train()
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
        if args.use_ema:
            ema_unet.to(accelerator.device)

        # Recalculate steps after prepare
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Trackers
        if accelerator.is_main_process:
            accelerator.init_trackers("text2image-finetune-sdxl-latent", config=vars(args))

        # Unwrap helper
        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        # ====== Loss logging & plotting ======
        loss_csv = os.path.join(args.output_dir, "loss.csv")
        loss_png = os.path.join(args.output_dir, "loss.png")
        os.makedirs(args.output_dir, exist_ok=True)
        loss_steps, loss_vals = [], []
        loss_history_limit = int(os.environ.get("PLOT_LOSS_HISTORY_LIMIT", "100000"))

        def read_y_range(file_path="range.txt"):
            if not os.path.isfile(file_path):
                return None
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    parts = f.read().strip().split()
                    if len(parts) != 2:
                        return None
                    ymin, ymax = float(parts[0]), float(parts[1])
                    if ymin >= ymax:
                        return None
                    return (ymin, ymax)
            except Exception:
                return None

        def _save_plot():
            if not loss_steps:
                return
            y_range = read_y_range("range.txt")
            plt.figure(figsize=(8, 4.5), dpi=150)
            plt.plot(loss_steps, loss_vals, label="loss", linewidth=1.0)
            plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training Loss")
            plt.grid(True, linewidth=0.3); plt.legend(loc="best")
            if y_range is not None:
                plt.ylim(*y_range)
            plt.tight_layout(); plt.savefig(loss_png); plt.close()

        def _maybe_empty_cache():
            if accelerator.device.type != "cuda":
                return
            try:
                free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            except Exception:
                return
            if free_bytes < 2.5 * 1024**3:
                torch.cuda.empty_cache()

        # ====== Preview helper ======
        preview_pipe = None
        preview_compel = None
        preview_empty_conditioning = None

        @torch.no_grad()
        def save_preview(step: int, prompt: str, height: Optional[int] = None, width: Optional[int] = None):
            nonlocal preview_pipe, preview_compel, preview_empty_conditioning
            if not prompt:
                return
            prompt = _escape_prompt_weight_syntax(prompt)
            negative_text = _escape_prompt_weight_syntax(getattr(args, "preview_negative", ""))
            preview_height = int(height) if height else args.resolution_height
            preview_width = int(width) if width else args.resolution_width
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            try:
                if preview_pipe is None:
                    vae_dtype = vae.dtype if hasattr(vae, "dtype") else torch.float32
                    preview_pipe = StableDiffusionXLPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        unet=unwrap_model(unet),
                        text_encoder=unwrap_model(text_encoder_one),
                        text_encoder_2=unwrap_model(text_encoder_two),
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    ).to(accelerator.device)
                    preview_pipe.vae.to(accelerator.device, dtype=vae_dtype)
                    preview_pipe.set_progress_bar_config(disable=True)
                else:
                    preview_pipe.unet = unwrap_model(unet)
                    if args.train_text_encoder:
                        preview_pipe.text_encoder = unwrap_model(text_encoder_one)
                        preview_pipe.text_encoder_2 = unwrap_model(text_encoder_two)
                    preview_pipe.vae = vae
                    preview_pipe.vae.to(accelerator.device, dtype=vae.dtype if hasattr(vae, "dtype") else torch.float32)

                if preview_compel is None:
                    preview_compel, preview_empty_conditioning = get_compel_for_sdxl(
                        [preview_pipe.tokenizer, preview_pipe.tokenizer_2],
                        [preview_pipe.text_encoder, preview_pipe.text_encoder_2],
                        device=accelerator.device,
                    )

                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = preview_compel([prompt])
                    negative_prompt_embeds, negative_pooled_prompt_embeds = preview_compel([negative_text])
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                    ) = preview_compel.pad_conditioning_tensors_to_same_length(
                        [prompt_embeds, negative_prompt_embeds], precomputed_padding=preview_empty_conditioning
                    )

                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                negative_prompt_embeds = negative_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)

                generator = (
                    torch.Generator(device=accelerator.device).manual_seed(args.preview_seed)
                    if args.preview_seed is not None
                    else None
                )
                add_time_ids = torch.tensor(
                    [
                        [
                            preview_height,
                            preview_width,
                            0,
                            0,
                            preview_height,
                            preview_width,
                        ]
                    ],
                    device=accelerator.device,
                    dtype=weight_dtype,
                )

                out = preview_pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=int(args.preview_steps),
                    guidance_scale=float(args.preview_scale),
                    width=preview_width,
                    height=preview_height,
                    generator=generator,
                    added_cond_kwargs={"time_ids": add_time_ids},
                ).images[0]
                out_dir = os.path.join(args.output_dir, "preview")
                os.makedirs(out_dir, exist_ok=True)
                out.save(os.path.join(out_dir, f"step_{step:08d}.png"))
            finally:
                if args.use_ema:
                    ema_unet.restore(unet.parameters())

        # ====== Text encode helper ======
        if args.train_text_encoder:
            def encode_texts_runtime(texts):
                return _encode_texts_with_trainable_encoders(texts)

        else:
            compel_obj_runtime, _ = get_compel_for_sdxl(
                [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two], text_encoder_one.device
            )

            def encode_texts_runtime(texts):
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = compel_obj_runtime(texts)
                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                return prompt_embeds, pooled_prompt_embeds

        # ====== Train (latent path) ======
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        logger.info("***** Running training (latent) *****")
        logger.info(f"  Num examples = {len(latent_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        global_step = 0
        first_epoch = 0

        # resume (built-in accelerate)
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            disable=not accelerator.is_local_main_process,
        )

        VARIANT_PROBS = np.array([0.4, 0.2, 0.1, 0.3], dtype=np.float64)

        for epoch in range(first_epoch, args.num_train_epochs):
            train_loss = 0.0
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    model_input = batch["model_input"].to(accelerator.device)
                    noise = torch.randn_like(model_input)
                    if args.noise_offset:
                        noise += args.noise_offset * torch.randn(
                            (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                        )
                    bsz = model_input.shape[0]
                    if args.timestep_bias_strategy == "none":
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                    else:
                        weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(model_input.device)
                        timesteps = torch.multinomial(weights, bsz, replacement=True).long()
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps).to(dtype=weight_dtype)

                    # time ids (all 1024 and no crop)
                    add_time_ids = torch.tensor(
                        [
                            [
                                args.resolution_height,
                                args.resolution_width,
                                0,
                                0,
                                args.resolution_height,
                                args.resolution_width,
                            ]
                        ],
                        device=accelerator.device,
                        dtype=weight_dtype,
                    ).repeat(bsz, 1)

                    # Build dynamic prompts per-sample
                    general_tags = batch["general"]
                    rating_tags = batch["rating"]
                    meta_tags = batch["meta"]
                    year_tags = batch["year"]
                    character_tags = batch["character"]
                    artist_tags = batch["artist"]
                    copyright_tags = batch["copyright"]
                    group_tags = batch.get("group") or []
                    type_tags = batch.get("type") or []
                    chosen = []
                    for i in range(bsz):
                        text = generate_variants_with_nl_list(
                            _split_clean_comma_list(general_tags[i]),
                            _normalize_artist_tags(_split_clean_comma_list(artist_tags[i])),
                            k=1,
                            token_budget=200,
                            head_keep=random.choices([12,14,16],[0.5,0.35,0.15])[0],
                            dropout=0.2,
                            max_general_per_variant=50,
                            characters=_split_clean_comma_list(character_tags[i]),
                            ratings=_split_clean_comma_list(rating_tags[i]),
                            years=_split_clean_comma_list(year_tags[i]),
                            groups=_split_clean_comma_list(group_tags[i] if i < len(group_tags) else ""),
                            cfg_dropout=args.proportion_empty_prompts,
                            type=type_tags[i] if i < len(type_tags) else "",
                        )
                        chosen.append(text[0])

                    prompt_embeds, pooled_prompt_embeds = encode_texts_runtime(chosen)

                    unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds}
                    model_pred = unet(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs=unet_added_conditions,
                        return_dict=False,
                    )[0]

                    if args.prediction_type is not None:
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    elif noise_scheduler.config.prediction_type == "sample":
                        target = model_input
                        model_pred = model_pred - noise
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if args.snr_gamma is None:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    else:
                        snr = compute_snr(noise_scheduler, timesteps)
                        mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                        if noise_scheduler.config.prediction_type == "epsilon":
                            mse_loss_weights = mse_loss_weights / snr
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            mse_loss_weights = mse_loss_weights / (snr + 1)
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = unet.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    if accelerator.is_main_process:
                        loss_steps.append(global_step)
                        loss_vals.append(train_loss)
                        if len(loss_steps) > loss_history_limit:
                            loss_steps.pop(0); loss_vals.pop(0)
                        # CSV logging temporarily disabled; keep data in memory for plotting only.
                        if args.plot_interval and (global_step % int(args.plot_interval) == 0):
                            _save_plot()
                        if args.preview_save_steps and (global_step % int(args.preview_save_steps) == 0 or global_step == 1):
                            # Use a random preview prompt from current batch
                            prev_prompt = random.choice(chosen) if chosen else ""
                            prev_prompts = generate_variants_with_nl_list(
                                _split_clean_comma_list(general_tags[0]),
                                _normalize_artist_tags(_split_clean_comma_list(artist_tags[0])),
                                k=2,
                                token_budget=200,
                                head_keep=random.choices([12,14,16],[0.5,0.35,0.15])[0],
                                dropout=0.2,
                                max_general_per_variant=50,
                                characters=_split_clean_comma_list(character_tags[0]),
                                ratings=_split_clean_comma_list(rating_tags[0]),
                                years=_split_clean_comma_list(year_tags[0]),
                                groups=_split_clean_comma_list(group_tags[0] if group_tags else ""),
                            )
                            for i, prev_prompt in enumerate(prev_prompts):
                                save_preview(global_step + i, prev_prompt)
                                os.makedirs(os.path.join(args.output_dir, "preview"), exist_ok=True)
                                with open(os.path.join(args.output_dir, "preview", f"prompt-{global_step + i}.txt"), "w") as f:
                                    f.write(prev_prompt)
                    train_loss = 0.0
                    if global_step % 10 == 0:
                        _maybe_empty_cache()

                    # checkpointing
                    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]
                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                if global_step >= args.max_train_steps:
                    break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Save pipeline
            u = unwrap_model(unet)
            if args.use_ema:
                ema_unet.copy_to(u.parameters())
            text_encoder_kwargs = {}
            if args.train_text_encoder:
                te1 = unwrap_model(text_encoder_one).to(device=torch.device("cpu"), dtype=torch.float32)
                te2 = unwrap_model(text_encoder_two).to(device=torch.device("cpu"), dtype=torch.float32)
                text_encoder_kwargs.update({"text_encoder": te1, "text_encoder_2": te2})
            vae_local = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipe = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=u,
                vae=vae_local,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
                **text_encoder_kwargs,
            )
            if args.prediction_type is not None:
                scheduler_args = {"prediction_type": args.prediction_type}
                pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config, **scheduler_args)
            pipe.save_pretrained(args.output_dir)

        accelerator.end_training()
        return
    bucket_map = None
    if not using_latent_db:
        # Branch B: raw image dataset via datasets/imagefolder
        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        resolution_sets_config = getattr(args, "resolution_sets_config", None)
        if args.dataset_name is not None and resolution_sets_config:
            raise ValueError("`--resolution_sets` cannot be combined with `--dataset_name`.")
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, data_dir=args.train_data_dir
            )
        else:
            dataset = None
            exclude_word_list = [
                "no_humans",
                "chibi",
                "character_profile",
                "lineart",
                "sketch",
                "monochrome",
                "comic",
                "text_focus",
                "1990s",
                "1980s",
                "retro_artstyle",
                "abstract",
            ]
            exclude_artist_entries = _load_filter_list("exclude_artists.txt")
            exclude_danbooru_entries = _load_filter_list("exclude_danbooru_artists.txt")
            exclude_source_id_entries = _load_filter_list("exclude_source_ids.txt")
            include_source_id_entries = _load_filter_list("include_source_ids.txt")
            exclude_artist_set = _build_filter_name_set(exclude_artist_entries)
            exclude_danbooru_set = _build_filter_name_set(exclude_danbooru_entries)
            exclude_source_id_set = {entry for entry in exclude_source_id_entries if entry}
            include_source_id_set = {entry for entry in include_source_id_entries if entry}

            if resolution_sets_config:
                per_resolution = []
                for idx, cfg in enumerate(resolution_sets_config):
                    ds = _load_and_filter_index_dataset(
                        cfg["index_file"],
                        cfg["train_data_dir"],
                        args.cache_dir,
                        exclude_word_list,
                        exclude_artist_set,
                        exclude_danbooru_set,
                        exclude_source_id_set,
                        include_source_id_set,
                        (args.seed if args.seed is not None else 42) + idx,
                    )
                    ds["train"] = ds["train"].map(
                        functools.partial(
                            _annotate_target_size,
                            height=cfg["resolution_height"],
                            width=cfg["resolution_width"],
                            label=cfg["label"],
                        ),
                        desc=f"Annotating resolution {cfg['resolution_height']}x{cfg['resolution_width']}",
                    )
                    per_resolution.append(ds["train"])
                merged = concatenate_datasets(per_resolution) if len(per_resolution) > 1 else per_resolution[0]
                dataset = datasets.DatasetDict({"train": merged})
            elif args.train_data_dir is not None or args.index_file is not None:
                index_path = args.index_file if args.index_file else os.path.join(args.train_data_dir, "index.jsonl")
                if os.path.isfile(index_path):
                    dataset = _load_and_filter_index_dataset(
                        index_path,
                        args.train_data_dir,
                        args.cache_dir,
                        exclude_word_list,
                        exclude_artist_set,
                        exclude_danbooru_set,
                        exclude_source_id_set,
                        include_source_id_set,
                        args.seed,
                    )
                    dataset["train"] = dataset["train"].map(
                        functools.partial(
                            _annotate_target_size,
                            height=args.resolution_height,
                            width=args.resolution_width,
                            label="default",
                        ),
                        desc="Annotating default resolution",
                    )

            if dataset is None:
                data_files = {}
                if args.train_data_dir is not None:
                    data_files["train"] = os.path.join(args.train_data_dir, "**")
                dataset = load_dataset(
                    "imagefolder",
                    data_files=data_files,
                    cache_dir=args.cache_dir,
                )
                dataset["train"] = dataset["train"].map(
                    functools.partial(
                        _annotate_target_size,
                        height=args.resolution_height,
                        width=args.resolution_width,
                        label="default",
                    ),
                    desc="Annotating default resolution",
                )

        if "target_height" not in dataset["train"].column_names or "target_width" not in dataset["train"].column_names:
            dataset["train"] = dataset["train"].map(
                functools.partial(
                    _annotate_target_size,
                    height=args.resolution_height,
                    width=args.resolution_width,
                    label="default",
                ),
                desc="Annotating default resolution",
            )

        bucket_map = None
        if resolution_sets_config:
            heights = dataset["train"]["target_height"]
            widths = dataset["train"]["target_width"]
            bucket_map = defaultdict(list)
            for idx, (h, w) in enumerate(zip(heights, widths)):
                bucket_map[(int(h), int(w))].append(idx)
            bucket_map = {k: v for k, v in bucket_map.items() if len(v) >= args.train_batch_size}

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        if dataset_columns is not None:
            image_column = dataset_columns[0]
        elif "image" in column_names:
            image_column = "image"
        else:
            image_column = column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    args.image_column = image_column

    metadata_column_map = {
        "general": "general",
        "rating": "rating",
        "meta": "meta",
        "year": "year",
        "character": "character",
        "artist": "artist",
        "group": "group",
        "type": "type",
    }
    metadata_column_map = {
        key: value for key, value in metadata_column_map.items() if value in column_names
    }
    if "general" not in metadata_column_map:
        metadata_column_map = None

    if args.caption_column is None:
        if dataset_columns is not None and len(dataset_columns) > 1:
            caption_column = dataset_columns[1]
        elif "text" in column_names:
            caption_column = "text"
        elif metadata_column_map and "general" in metadata_column_map:
            caption_column = metadata_column_map["general"]
        else:
            caption_column = None
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            if metadata_column_map and "general" in metadata_column_map:
                caption_column = metadata_column_map["general"]
            else:
                raise ValueError(
                    f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )

    if caption_column is None and metadata_column_map is None:
        raise ValueError("Could not determine a caption column or metadata columns for prompt generation.")

    args.caption_column = caption_column

    metadata_aliases = list(metadata_column_map.keys()) if metadata_column_map else []

    # Preprocessing the datasets.
    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
    if interpolation is None:
        raise ValueError(f"Unsupported interpolation mode {interpolation=}.")
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def preprocess_train(examples):
        processed_images = []
        for image in examples[image_column]:
            img = image
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
            canvas.alpha_composite(img)
            processed_images.append(canvas.convert("RGB"))
        images = processed_images
        # image aug
        target_heights = examples.get("target_height")
        target_widths = examples.get("target_width")
        if target_heights is None:
            target_heights = [args.resolution_height] * len(images)
        if target_widths is None:
            target_widths = [args.resolution_width] * len(images)

        original_sizes = []
        all_images = []
        crop_top_lefts = []
        target_sizes = []

        for idx, image in enumerate(images):
            original_sizes.append((image.height, image.width))
            target_h = int(target_heights[idx])
            target_w = int(target_widths[idx])
            resize_size = (target_h, target_w)
            scale = max(
                target_h / image.height,
                target_w / image.width,
            )
            resized_height = max(target_h, int(math.ceil(image.height * scale)))
            resized_width = max(target_w, int(math.ceil(image.width * scale)))
            if resized_height != image.height or resized_width != image.width:
                image = resize(
                    image,
                    [resized_height, resized_width],
                    interpolation=interpolation,
                    antialias=True,
                )
            if args.random_flip and random.random() < 0.5:
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - target_h) / 2.0)))
                x1 = max(0, int(round((image.width - target_w) / 2.0)))
                image = crop(image, y1, x1, target_h, target_w)
            else:
                y1, x1, h, w = transforms.RandomCrop.get_params(image, resize_size)
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            target_sizes.append((target_h, target_w))
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["target_sizes"] = target_sizes
        examples["pixel_values"] = all_images
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    using_precomputed_dataset = args.precompute_text_embeddings and args.precompute_vae_latents

    if using_precomputed_dataset:
        # Let's first compute all the embeddings so that we can free up the text encoders
        # from memory. We will pre-compute the VAE encodings too.
        text_encoders = [text_encoder_one, text_encoder_two]
        tokenizers = [tokenizer_one, tokenizer_two]
        compute_embeddings_fn = functools.partial(
            encode_prompt,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            proportion_empty_prompts=args.proportion_empty_prompts,
            caption_column=args.caption_column,
            metadata_columns=metadata_column_map,
        )
        compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
        with accelerator.main_process_first():
            from datasets.fingerprint import Hasher

            # fingerprint used by the cache for the other processes to load the result
            # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
            new_fingerprint = Hasher.hash(args)
            new_fingerprint_for_vae = Hasher.hash((vae_path, args))
            if accelerator.is_main_process:
                logger.info(
                    f"Precomputing prompt embeddings with batch size {args.text_encode_batch_size}"
                )
            train_dataset_with_embeddings = train_dataset.map(
                compute_embeddings_fn,
                batched=True,
                batch_size=args.text_encode_batch_size,
                new_fingerprint=new_fingerprint,
            )
            if accelerator.is_main_process:
                logger.info(
                    f"Precomputing VAE latents with batch size {args.vae_encode_batch_size}"
                )
            train_dataset_with_vae = train_dataset.map(
                compute_vae_encodings_fn,
                batched=True,
                batch_size=args.vae_encode_batch_size,
                new_fingerprint=new_fingerprint_for_vae,
            )
            columns_to_remove = [col for col in ["image", "text"] if col in train_dataset_with_vae.column_names]
            train_dataset_with_vae = (
                train_dataset_with_vae.remove_columns(columns_to_remove) if columns_to_remove else train_dataset_with_vae
            )
            # Drop duplicate metadata columns from the embeddings dataset so axis=1 concatenation keeps a single copy.
            shared_columns = set(train_dataset_with_embeddings.column_names).intersection(
                set(train_dataset_with_vae.column_names)
            )
            keep_columns = {"prompt_embeds", "pooled_prompt_embeds"}
            if args.image_column in train_dataset_with_embeddings.column_names:
                keep_columns.add(args.image_column)
            columns_to_remove_from_embeddings = [
                column for column in shared_columns if column not in keep_columns
            ]
            if columns_to_remove_from_embeddings:
                train_dataset_with_embeddings = train_dataset_with_embeddings.remove_columns(
                    columns_to_remove_from_embeddings
                )
            precomputed_dataset = concatenate_datasets(
                [train_dataset_with_embeddings, train_dataset_with_vae], axis=1
            )
            precomputed_dataset = precomputed_dataset.with_transform(preprocess_train)

        del compute_vae_encodings_fn, compute_embeddings_fn, text_encoder_one, text_encoder_two
        del text_encoders, tokenizers, vae
        gc.collect()
        if is_torch_npu_available():
            torch_npu.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        def collate_fn(examples):
            model_input = torch.stack([example["model_input"] if isinstance(example["model_input"], torch.Tensor) else torch.tensor(example["model_input"]) for example in examples])
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]
            prompt_embeds = torch.stack([example["prompt_embeds"] if isinstance(example["prompt_embeds"], torch.Tensor) else torch.tensor(example["prompt_embeds"]) for example in examples])
            pooled_prompt_embeds = torch.stack([example["pooled_prompt_embeds"] if isinstance(example["pooled_prompt_embeds"], torch.Tensor) else torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

            batch = {
                "model_input": model_input,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
                "target_sizes": target_sizes,
            }

            # Preserve metadata/caption fields for preview prompt reconstruction if present.
            for alias in metadata_aliases:
                column = metadata_column_map[alias]
                if column in examples[0]:
                    batch[alias] = [example.get(column) or "" for example in examples]
            if args.caption_column and args.caption_column in examples[0]:
                batch["_caption_texts"] = [example.get(args.caption_column) or "" for example in examples]

            return batch

        dataset_for_loader = precomputed_dataset
    else:
        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] if isinstance(example["pixel_values"], torch.Tensor) else torch.tensor(example["pixel_values"]) for example in examples])
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]

            batch = {
                "pixel_values": pixel_values,
                "original_sizes": original_sizes,
                "crop_top_lefts": crop_top_lefts,
                "target_sizes": target_sizes,
            }

            for alias in metadata_aliases:
                column = metadata_column_map[alias]
                if column in examples[0]:
                    batch[alias] = [example.get(column) or "" for example in examples]
            if args.caption_column and args.caption_column in examples[0]:
                batch["_caption_texts"] = [example.get(args.caption_column) or "" for example in examples]

            return batch

        dataset_for_loader = train_dataset

    # DataLoaders creation:
    batch_sampler = None
    if bucket_map:
        batch_sampler = ResolutionBucketBatchSampler(
            bucket_map, args.train_batch_size, shuffle=True, seed=args.seed or 0
        )

    if batch_sampler is not None:
        train_dataloader = torch.utils.data.DataLoader(
            dataset_for_loader,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            dataset_for_loader,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler)
        text_encoder_one.train()
        text_encoder_two.train()
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune-sdxl", config=vars(args))

    # Function for unwrapping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if torch.backends.mps.is_available() or "playground" in args.pretrained_model_name_or_path:
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset_for_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    preview_pipe = None
    preview_compel = None
    preview_empty_conditioning = None

    loss_csv = os.path.join(args.output_dir, "loss.csv")
    loss_png = os.path.join(args.output_dir, "loss.png")
    os.makedirs(args.output_dir, exist_ok=True)
    loss_steps, loss_vals = [], []
    loss_history_limit = int(os.environ.get("PLOT_LOSS_HISTORY_LIMIT", "100000"))

    def _read_y_range(file_path="range.txt"):
        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
                if len(parts) != 2:
                    return None
                ymin, ymax = float(parts[0]), float(parts[1])
                if ymin >= ymax:
                    return None
                return (ymin, ymax)
        except Exception:
            return None

    def _save_plot():
        if not loss_steps:
            return
        y_range = _read_y_range("range.txt")
        plt.figure(figsize=(8, 4.5), dpi=150)
        plt.plot(loss_steps, loss_vals, label="loss", linewidth=1.0)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.grid(True, linewidth=0.3)
        plt.legend(loc="best")
        if y_range is not None:
            plt.ylim(*y_range)
        plt.tight_layout()
        plt.savefig(loss_png)
        plt.close()

    def _maybe_empty_cache():
        if accelerator.device.type != "cuda":
            return
        try:
            free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        except Exception:
            return
        if free_bytes < 2.5 * 1024**3:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def save_preview(step: int, prompt: str, height: Optional[int] = None, width: Optional[int] = None):
        nonlocal preview_pipe, preview_compel, preview_empty_conditioning
        if not prompt:
            return
        prompt = _escape_prompt_weight_syntax(prompt)
        negative_text = _escape_prompt_weight_syntax(getattr(args, "preview_negative", ""))
        preview_height = int(height) if height else args.resolution_height
        preview_width = int(width) if width else args.resolution_width
        if args.use_ema:
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())
        try:
            if preview_pipe is None:
                vae_dtype = vae.dtype if hasattr(vae, "dtype") else torch.float32
                preview_pipe = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    tokenizer=tokenizer_one,
                    tokenizer_2=tokenizer_two,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                ).to(accelerator.device)
                preview_pipe.vae.to(accelerator.device, dtype=vae_dtype)
                preview_pipe.set_progress_bar_config(disable=True)
            else:
                preview_pipe.unet = unwrap_model(unet)
                if args.train_text_encoder:
                    preview_pipe.text_encoder = unwrap_model(text_encoder_one)
                    preview_pipe.text_encoder_2 = unwrap_model(text_encoder_two)
                preview_pipe.vae = vae
                preview_pipe.vae.to(accelerator.device, dtype=vae.dtype if hasattr(vae, "dtype") else torch.float32)

            if preview_compel is None:
                preview_compel, preview_empty_conditioning = get_compel_for_sdxl(
                    [preview_pipe.tokenizer, preview_pipe.tokenizer_2],
                    [preview_pipe.text_encoder, preview_pipe.text_encoder_2],
                    device=accelerator.device,
                )

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = preview_compel([prompt])
                negative_prompt_embeds, negative_pooled_prompt_embeds = preview_compel([negative_text])
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                ) = preview_compel.pad_conditioning_tensors_to_same_length(
                    [prompt_embeds, negative_prompt_embeds], precomputed_padding=preview_empty_conditioning
                )

            prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
            negative_prompt_embeds = negative_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)

            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.preview_seed)
                if args.preview_seed is not None
                else None
            )
            add_time_ids = torch.tensor(
                [
                    [
                        preview_height,
                        preview_width,
                        0,
                        0,
                        preview_height,
                        preview_width,
                    ]
                ],
                device=accelerator.device,
                dtype=weight_dtype,
            )

            result = preview_pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=int(args.preview_steps),
                guidance_scale=float(args.preview_scale),
                width=preview_width,
                height=preview_height,
                generator=generator,
                added_cond_kwargs={"time_ids": add_time_ids},
            )
            image = result.images[0]
            out_dir = os.path.join(args.output_dir, "preview")
            os.makedirs(out_dir, exist_ok=True)
            image.save(os.path.join(out_dir, f"step_{step:08d}.png"))
        finally:
            if args.use_ema:
                ema_unet.restore(unet.parameters())

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if not using_precomputed_dataset:
        if args.train_text_encoder:

            def encode_texts_runtime(texts):
                return _encode_texts_with_trainable_encoders(texts)

        else:
            runtime_compel, _ = get_compel_for_sdxl(
                [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two], accelerator.device
            )

            def encode_texts_runtime(texts):
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = runtime_compel(texts)
                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                return prompt_embeds, pooled_prompt_embeds

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Sample noise that we'll add to the latents
                if using_precomputed_dataset:
                    model_input = batch["model_input"].to(accelerator.device)
                else:
                    pixel_values = batch["pixel_values"].to(memory_format=torch.contiguous_format).float()
                    pixel_values = pixel_values.to(accelerator.device, dtype=vae.dtype)
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (latents * vae.config.scaling_factor).to(weight_dtype)
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )

                bsz = model_input.shape[0]
                if args.timestep_bias_strategy == "none":
                    # Sample a random timestep for each image without bias.
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                else:
                    # Sample a random timestep for each image, potentially biased by the timestep weights.
                    # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
                    weights = generate_timestep_weights(args, noise_scheduler.config.num_train_timesteps).to(
                        model_input.device
                    )
                    timesteps = torch.multinomial(weights, bsz, replacement=True).long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps).to(dtype=weight_dtype)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left, target_size):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [
                        compute_time_ids(s, c, t)
                        for s, c, t in zip(batch["original_sizes"], batch["crop_top_lefts"], batch["target_sizes"])
                    ]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                if using_precomputed_dataset:
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device, dtype=weight_dtype)
                    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                else:
                    bsz = model_input.shape[0]
                    chosen_prompts = []
                    if metadata_column_map and "general" in batch:
                        general_tags = batch.get("general") or []
                        rating_tags = batch.get("rating") or []
                        meta_tags = batch.get("meta") or []
                        year_tags = batch.get("year") or []
                        character_tags = batch.get("character") or []
                        artist_tags = batch.get("artist") or []
                        group_tags = batch.get("group") or []
                        type_tags = batch.get("type") or []
                        for i in range(bsz):
                            variants = generate_variants_with_nl_list(
                                _split_clean_comma_list(general_tags[i] if i < len(general_tags) else ""),
                                _normalize_artist_tags(_split_clean_comma_list(artist_tags[i] if i < len(artist_tags) else "")),
                                k=1,
                                token_budget=200,
                                head_keep=random.choices([10, 12, 14], [0.5, 0.35, 0.15])[0],
                                dropout=0.2,
                                max_general_per_variant=50,
                                characters=_split_clean_comma_list(character_tags[i] if i < len(character_tags) else ""),
                                ratings=_split_clean_comma_list(rating_tags[i] if i < len(rating_tags) else ""),
                                years=_split_clean_comma_list(year_tags[i] if i < len(year_tags) else ""),
                                nl_texts=_split_clean_comma_list(meta_tags[i] if i < len(meta_tags) else ""),
                                groups=_split_clean_comma_list(group_tags[i] if i < len(group_tags) else ""),
                                cfg_dropout=args.proportion_empty_prompts,
                                type=type_tags[i] if i < len(type_tags) else "",
                            )
                            chosen_prompts.append(variants[0] if variants else "")
                    else:
                        captions = batch.get("_caption_texts") or []
                        for cap in captions:
                            if isinstance(cap, str):
                                chosen_prompts.append(cap)
                            elif isinstance(cap, (list, tuple)) and cap:
                                chosen_prompts.append(str(cap[0]))
                            else:
                                chosen_prompts.append("")
                    if not chosen_prompts:
                        chosen_prompts = [""] * bsz
                    prompt_embeds, pooled_prompt_embeds = encode_texts_runtime(chosen_prompts)
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                elif noise_scheduler.config.prediction_type == "sample":
                    # We set the target to latents here, but the model_pred will return the noise sample prediction.
                    target = model_input
                    # We will have to subtract the noise residual from the prediction to get the target sample.
                    model_pred = model_pred - noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://huggingface.co/papers/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.train_text_encoder:
                        params_to_clip = chain(
                            unet.parameters(), text_encoder_one.parameters(), text_encoder_two.parameters()
                        )
                    else:
                        params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    loss_steps.append(global_step)
                    loss_vals.append(train_loss)
                    if len(loss_steps) > loss_history_limit:
                        loss_steps.pop(0); loss_vals.pop(0)
                    # CSV logging temporarily disabled; retain values in memory for plotting only.
                    if args.plot_interval and (global_step % int(args.plot_interval) == 0):
                        _save_plot()
                train_loss = 0.0
                if global_step % 5 == 0:
                    _maybe_empty_cache()

                if accelerator.is_main_process and args.preview_save_steps:
                    if global_step % int(args.preview_save_steps) == 0 or global_step == 1:
                        preview_prompts: list[str] = []
                        def _first_str(values):
                            if not values:
                                return ""
                            value = values[0]
                            if isinstance(value, (list, tuple)):
                                return ", ".join(str(v) for v in value if v)
                            return str(value or "")

                        if metadata_column_map and "general" in metadata_column_map:
                            general_tags = batch.get("general") or []
                            artist_tags = batch.get("artist") or []
                            rating_tags = batch.get("rating") or []
                            year_tags = batch.get("year") or []
                            character_tags = batch.get("character") or []
                            meta_tags = batch.get("meta") or []
                            group_tags = batch.get("group") or []
                            type_tags = batch.get("type") or []
                            if general_tags:
                                preview_prompts = generate_variants_with_nl_list(
                                    _split_clean_comma_list(_first_str(general_tags)),
                                    _normalize_artist_tags(_split_clean_comma_list(_first_str(artist_tags))),
                                    k=2,
                                    token_budget=200,
                                    head_keep=random.choices([12,14,16],[0.5,0.35,0.15])[0],
                                    dropout=0.2,
                                    max_general_per_variant=50,
                                    characters=_split_clean_comma_list(_first_str(character_tags)),
                                    ratings=_split_clean_comma_list(_first_str(rating_tags)),
                                    years=_split_clean_comma_list(_first_str(year_tags)),
                                    nl_texts=_split_clean_comma_list(_first_str(meta_tags)),
                                    groups=_split_clean_comma_list(_first_str(group_tags)),
                                    type=_first_str(type_tags),
                                )
                        if not preview_prompts:
                            captions = batch.get("_caption_texts") or []
                            if captions and isinstance(captions[0], str) and captions[0].strip():
                                preview_prompts = [captions[0].strip()]

                        base_height = args.resolution_height
                        base_width = args.resolution_width
                        target_sizes = batch.get("target_sizes") or []
                        if target_sizes:
                            try:
                                base_height = int(target_sizes[0][0])
                                base_width = int(target_sizes[0][1])
                            except Exception:
                                base_height = args.resolution_height
                                base_width = args.resolution_width

                        for idx, preview_prompt in enumerate(preview_prompts):
                            save_preview(global_step + idx, preview_prompt, height=base_height, width=base_width)
                            out_dir = os.path.join(args.output_dir, "preview")
                            try:
                                with open(
                                    os.path.join(out_dir, f"prompt-{global_step + idx}.txt"),
                                    "w",
                                    encoding="utf-8",
                                ) as f:
                                    f.write(preview_prompt)
                            except Exception:
                                pass

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                # create pipeline
                vae = AutoencoderKL.from_pretrained(
                    vae_path,
                    subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                    revision=args.revision,
                    variant=args.variant,
                )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    unet=accelerator.unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    **(
                        {
                            "text_encoder": accelerator.unwrap_model(text_encoder_one),
                            "text_encoder_2": accelerator.unwrap_model(text_encoder_two),
                        }
                        if args.train_text_encoder
                        else {}
                    ),
                )
                if args.prediction_type is not None:
                    scheduler_args = {"prediction_type": args.prediction_type}
                    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = (
                    torch.Generator(device=accelerator.device).manual_seed(args.seed)
                    if args.seed is not None
                    else None
                )
                pipeline_args = {"prompt": args.validation_prompt}

                with autocast_ctx:
                    images = [
                        pipeline(**pipeline_args, generator=generator, num_inference_steps=25).images[0]
                        for _ in range(args.num_validation_images)
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                if is_torch_npu_available():
                    torch_npu.npu.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if args.train_text_encoder:
                    text_encoder_one.train()
                    text_encoder_two.train()

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        text_encoder_kwargs = {}
        if args.train_text_encoder:
            te1 = unwrap_model(text_encoder_one).to(device=torch.device("cpu"), dtype=torch.float32)
            te2 = unwrap_model(text_encoder_two).to(device=torch.device("cpu"), dtype=torch.float32)
            text_encoder_kwargs.update({"text_encoder": te1, "text_encoder_2": te2})

        # Serialize pipeline.
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
            **text_encoder_kwargs,
        )
        if args.prediction_type is not None:
            scheduler_args = {"prediction_type": args.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline = pipeline.to(accelerator.device)
            generator = (
                torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
            )

            with autocast_ctx:
                images = [
                    pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                    for _ in range(args.num_validation_images)
                ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id=repo_id,
                images=images,
                validation_prompt=args.validation_prompt,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
