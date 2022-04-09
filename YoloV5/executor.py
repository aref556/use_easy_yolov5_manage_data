from jina import Executor, DocumentArray, requests


class YoloV5(Executor):
    """"""
    @requests
    def foo(self, docs: DocumentArray, **kwargs):
        pass
