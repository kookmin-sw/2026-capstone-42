from __future__ import annotations

import pickle
import zipfile
from typing import List


class CitationGraphPKL:
    """citation_graph.zip 내부의 citation_graph.pkl을 로드해 CitationGraph Protocol을 만족하는 wrapper."""

    def __init__(self, zip_path: str) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("citation_graph.pkl") as f:
                data = pickle.load(f)
        self._forward = data["forward"]    # paper_id → List[paper_id] (이 논문이 인용한 논문들)
        self._backward = data["backward"]  # paper_id → List[paper_id] (이 논문을 인용한 논문들)

    def references(self, paper_id: str) -> List[str]:
        return self._forward.get(paper_id, [])

    def cited_by(self, paper_id: str) -> List[str]:
        return self._backward.get(paper_id, [])
