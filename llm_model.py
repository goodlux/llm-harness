# llmharness/llm_model.py
from dataclasses import dataclass,field
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import litellm
from llmharness.harness import LLMHarness
import asyncio

@dataclass
class LLMModel:
    model_name: str
    instance_name: Optional[str] = None
    llm_params: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    rate_limit: float = 2.0  # requests per second
    _last_request: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.instance_name is None:
            self.instance_name = self.model_name
        if self.llm_params is None:
            self.llm_params = {}
        if self.conversation_history is None:
            self.conversation_history = []

    async def complete(self, harness: LLMHarness, prompt: str, **kwargs) -> str:
        # Apply rate limiting
        elapsed = time.time() - self._last_request
        if elapsed < (1 / self.rate_limit):
            await asyncio.sleep((1 / self.rate_limit) - elapsed)

        params: Dict[str, Any] = {
            **(self.llm_params or {}),
            **kwargs
        }

        response = await harness.complete(
            model=self.model_name,
            prompt=prompt,
            **params
        )

        self._last_request = time.time()
        self.conversation_history.append({
            "instance": self.instance_name,
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

        return response
