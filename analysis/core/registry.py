from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol

from .types import StepExtractContext


class StepExtractor(Protocol):
    extractor_id: str

    def __call__(self, context: "ExtractionRuntimeContext", payload: Dict) -> Dict:
        ...


@dataclass
class ExtractionRuntimeContext:
    step_context: StepExtractContext
    cfg: Any
    society: Any
    data_payload: Dict[str, Any]
    eval_context: Any
    store: Any


class ExtractorRegistry:
    def __init__(self):
        self._extractors: Dict[str, Callable[[ExtractionRuntimeContext, Dict], Dict]] = {}

    def register(self, extractor_id: str, fn: Callable[[ExtractionRuntimeContext, Dict], Dict]) -> None:
        if extractor_id in self._extractors:
            raise ValueError(f"Extractor already registered: {extractor_id}")
        self._extractors[extractor_id] = fn

    def list_ids(self) -> List[str]:
        return list(self._extractors.keys())

    def run_all(self, context: ExtractionRuntimeContext, payload: Dict) -> Dict:
        output = payload
        for extractor_id in self.list_ids():
            output = self._extractors[extractor_id](context, output)
        return output
