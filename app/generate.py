"""vLLM client — calls the OpenAI-compatible chat completions endpoint."""
import httpx


class Generator:
    def __init__(
        self,
        model: str,
        vllm_host: str,
        vllm_port: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        system_prompt: str,
    ):
        self.model = model
        self.base_url = f"http://{vllm_host}:{vllm_port}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, question: str, context: str) -> str:
        prompt = self.system_prompt.replace("{context}", context)
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def translate_to_english(self, question: str) -> str:
        """Translate a non-English question to English for retrieval.

        The documentation is in English, so the embedding model (English-only)
        needs an English query to retrieve relevant chunks.
        """
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's question to English. "
                            "Output ONLY the translated question, nothing else. "
                            "If the question is already in English, return it as-is."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "top_p": 1.0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    async def is_ready(self) -> bool:
        try:
            resp = await self.client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
