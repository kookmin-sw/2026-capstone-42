from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import numpy as np


@dataclass
class RetrievalHit:
    paper_id: str
    faiss_score: float
    rank: int


@dataclass
class PaperMetadata:
    year: int | None
    citation_count: int | None
    venue: str | None = None


class CitationGraph(Protocol):
    def references(self, paper_id: str) -> List[str]: ...
    def cited_by(self, paper_id: str) -> List[str]: ...


Embeddings = Dict[str, np.ndarray]
BibEmbeddings = List[np.ndarray]
Metadata = Dict[str, PaperMetadata]
