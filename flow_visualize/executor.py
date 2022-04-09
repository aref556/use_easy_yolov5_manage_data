from typing import Optional, Iterable
from collections import defaultdict

import numpy as np

from jina import Document, DocumentArray, Executor, requests


_ALLOWED_METRICS = ['min', 'max', 'mean_min', 'mean_max']
DEFAULT_FPS = 1

#  =========
class VideoLoaderTest(Executor):
    def __init__(self,
                 modality: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            chunks = filter(lambda d: d.modality == self.modality, doc.chunks)
            doc.chunks = chunks
        return docs

class PersonDetectionTest(Executor):
    def __init__(self,
                 modality: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            chunks = filter(lambda d: d.modality == self.modality, doc.chunks)
            doc.chunks = chunks
        return docs

class PreprocessingDataTest(Executor):
    def __init__(self,
                 modality: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            chunks = filter(lambda d: d.modality == self.modality, doc.chunks)
            doc.chunks = chunks
        return docs

class PersonReidentificationTest(Executor):
    def __init__(self,
                 modality: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            chunks = filter(lambda d: d.modality == self.modality, doc.chunks)
            doc.chunks = chunks
        return docs

class SimpleIndexerTest(Executor):
    def __init__(self,
                 modality: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            chunks = filter(lambda d: d.modality == self.modality, doc.chunks)
            doc.chunks = chunks
        return docs

class ImageLoaderTest(Executor):
    def __init__(self,
                 modality: str = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = modality

    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            chunks = filter(lambda d: d.modality == self.modality, doc.chunks)
            doc.chunks = chunks
        return docs