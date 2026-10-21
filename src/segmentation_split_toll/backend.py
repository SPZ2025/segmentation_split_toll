"""Lazy E5 tokenizer and encoder; passthrough never loads model weights."""
from __future__ import annotations

import numpy as np


MODEL_ID = 'intfloat/multilingual-e5-small'
MODEL_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'


class ModelUnavailableError(RuntimeError):
    pass


class E5Backend:
    def __init__(self, *, model_path=None, local_files_only=True, device=None, batch_size=32):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.model_path = str(model_path) if model_path is not None else MODEL_ID
        self.revision = None if model_path is not None else MODEL_REVISION
        self.local_files_only = bool(local_files_only)
        self.device, self.batch_size = device, batch_size
        self._tokenizer = self._encoder = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, revision=self.revision,
                    local_files_only=self.local_files_only, use_fast=True)
            except OSError as exc:
                raise ModelUnavailableError('E5 tokenizer unavailable. Use --download-model once, or --model-path with a local E5 directory.') from exc
        return self._tokenizer

    def count_many(self, texts):
        return [int(n) for n in self.tokenizer(list(texts), add_special_tokens=False,
            truncation=False, padding=False, return_length=True, return_attention_mask=False,
            return_token_type_ids=False, verbose=False)['length']]

    def count(self, text):
        return self.count_many([text])[0]

    def encode(self, texts):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._encoder = SentenceTransformer(self.model_path, revision=self.revision,
                    local_files_only=self.local_files_only, device=self.device)
            except OSError as exc:
                raise ModelUnavailableError('E5 weights unavailable. Use --download-model or --model-path.') from exc
        tokenizer = self.tokenizer
        prefix = 'query: '
        overhead = len(tokenizer.encode(prefix, add_special_tokens=True))
        budget = int(self._encoder.max_seq_length) - overhead - 4
        if budget < 32:
            raise ValueError('Unexpected E5 sequence length')
        inputs, assignment = [], []
        for index, text in enumerate(texts):
            ids = tokenizer.encode(text.strip(), add_special_tokens=False, truncation=False)
            if not ids:
                inputs.append(prefix)
                assignment.append((index, 1))
                continue
            for offset in range(0, len(ids), budget):
                chunk = ids[offset:offset + budget]
                inputs.append(prefix + tokenizer.decode(chunk, skip_special_tokens=True))
                assignment.append((index, len(chunk)))
        encoded = self._encoder.encode(inputs, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)
        pooled = np.zeros((len(texts), encoded.shape[1]), dtype=np.float64)
        weights = np.zeros(len(texts))
        for vector, (index, weight) in zip(encoded, assignment):
            pooled[index] += vector * weight
            weights[index] += weight
        pooled /= weights[:, None]
        return (pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)).astype(np.float32)
