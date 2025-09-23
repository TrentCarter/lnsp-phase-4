# src/search_backends/stub.py
class StubSearcher:
    def topk(self, qvec, k=5):
        return [{"id": "S1", "score": 0.99}]
