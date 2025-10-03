# coding=utf-8
# Copyright 2025 NUS Show Lab.
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

from datasets import load_dataset
import collections
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets_showo.utils import image_transform, format_interleaved_sequence
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class HQEditDataset(Dataset):
    """Dataset for High-Quality Image Editing (HQEdit) with interleaved image-text pairs."""

    def __init__(
            self,
            root: str,
            anno_path: str,
            text_tokenizer: Any,
            max_seq_len: int = 3840,
            image_size: int = 384,
            latent_height: int = 24,
            latent_width: int = 24,
            num_image_tokens: int = 576,
            cond_dropout_prob: float = 0.1,
            max_num_images: int = 4,
            loader: Callable[[str], Any] = default_loader,
            showo_token_ids: Optional[Dict[str, int]] = None,
            system: Tuple[str, str, str] = ("", "", ""),
            min_res: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initializes the VIST dataset.

        Args:
            root: Root directory of images.
            text_tokenizer: Tokenizer for text processing.
            max_seq_len: Maximum sequence length.
            image_size: Size to which images are resized.
            latent_height: Height of latent representation.
            latent_width: Width of latent representation.
            num_image_tokens: Number of tokens representing an image.
            cond_dropout_prob: Probability of conditioning dropout.
            max_num_images: Maximum number of images per sample.
            loader: Function to load an image given its path.
            anno_path: Path to the annotation JSON file.
            showo_token_ids: Dictionary of special token IDs.
            system: Tuple of system prompt strings.
            min_res: Minimum resolution (height, width) for images.
        """
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.data_type = "interleaved_data"
        self.transform = image_transform
        self.max_num_images = max_num_images

        self.root = root
        self.anno_path = anno_path
        self.samples: List[Dict[str, Any]] = []
        self.loader = loader
        self.dataset = load_dataset("/mnt/pentagon/dataset/HQ-Edit", split="train")
        print(f"HE Edit dataset loaded. {len(self.dataset)} samples!")

        self.flag_tokens = self.text_tokenizer(
            "", add_special_tokens=False
        ).input_ids
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)

        if len(self.system_tokens[0]) == 0:
            # 4 for bos, eos, boi, and eoi tokens
            self.max_text_len = (
                                        max_seq_len
                                        - len(self.flag_tokens)
                                        - (num_image_tokens + 2) * max_num_images
                                        - 2
                                ) // max_num_images
        else:
            # 4 for bos, eos, boi, and eoi tokens
            # 1 for eos after text token (a bit tricky)
            # see more details in def format_sequence_gen_qwen2_5(...)
            self.max_text_len = (
                                        max_seq_len
                                        - (num_image_tokens + 2) * max_num_images
                                        - 2
                                        - self.system_token_len
                                        - 1
                                ) // max_num_images

        self.min_res = min_res if min_res is not None else (256, 256)

    def _get_interleaved_data(
            self, anno: Dict[str, Any]
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[List[int]]], List[str]]:
        
        image_paths = [anno['input_image'], anno['output_image']]
        texts = ["", anno["edit"]]

        image_list: List[Optional[torch.Tensor]] = []
        text_token_list: List[Optional[List[int]]] = []
        for path, text in zip(image_paths, texts):
            image = self.transform(path, resolution=self.image_size)
            image_list.append(image)

            text_tokens = self.text_tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_len,
            ).input_ids
            text_token_list.append(text_tokens)

        # Add flag token to the first text token list
        text_token_list[0] = self.flag_tokens + text_token_list[0]

        # Pad lists if fewer than max_num_images
        if len(image_list) != self.max_num_images:
            image_list += [None] * (self.max_num_images - len(image_list))
            text_token_list += [None] * (self.max_num_images - len(text_token_list))
            texts += [''] * (self.max_num_images - len(texts))

        return image_list, text_token_list, texts

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        anno = self.dataset[idx]
        image_list, text_token_list, texts = self._get_interleaved_data(anno)
        (
                text_tokens,
                text_labels,
                modality_positions,
                text_mask,
                image_mask,
        ) = format_interleaved_sequence(
                image_list,
                text_token_list,
                self.bos_id,
                self.eos_id,
                self.boi_id,
                self.eoi_id,
                self.pad_id,
                self.img_pad_id,
                self.num_image_tokens,
                self.max_seq_len,
                self.max_num_images,
        )

        # Ignore flag tokens in the label (first one is bos token)
        # text_labels[1: len(self.flag_tokens) + 1] = -100

        temp: List[torch.Tensor] = []
        for img in image_list:
                if img is not None:
                    temp.append(img)
                else:
                    temp.append(torch.zeros((3, self.image_size, self.image_size)))

        image = torch.stack(temp, dim=0)
        return {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': image,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
                'texts': texts,
                'data_type': self.data_type,
            }



    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data."""
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)
        for key, value in batched.items():
            if key not in ('texts', 'data_type'):
                batched[key] = torch.stack(value, dim=0)
        return batched
