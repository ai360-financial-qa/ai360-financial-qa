import asyncio
from typing import List
from tqdm import auto as tqdm
from openai import AsyncOpenAI
from financial_qa.base import BaseChunker, Chunk


class SummaryChunker(BaseChunker):
    """
    Generates synthetic summary chunks over groups of original chunks
    using an LLM via OpenRouter asynchronously.
    """

    def __init__(
            self,
            base_chunker: BaseChunker,
            api_key: str = "",
            model: str = 'google/gemini-2.0-flash-lite-001',
            window_size: int = 5,
            temperature: float = 0.0,
            max_retries: int = 3,
            max_tokens: int = 400,
            max_concurrent: int = 8,
            include_source_refs: bool = False,
            prompt_template: str = None,
    ):
        self.base_chunker = base_chunker
        self.window_size = window_size
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.include_source_refs = include_source_refs

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai",
            api_key=api_key,
        )

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = (
                "You are an expert financial analyst specializing in banking and regulatory reporting.\n"
                "Your task is to synthesize the provided text fragments into a comprehensive, high-density summary.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Focus heavily on quantitative data, key metrics, financial indicators, and corporate actions mentioned.\n"
                "2. Maintain strict factual consistency. Do NOT assume, extrapolate, or introduce external knowledge.\n"
                "3. If facts contradict each other, state the contradiction clearly.\n"
                "4. Rely ONLY on the clear facts directly mentioned in the text.\n\n"
                "OUTPUT FORMAT:\n"
                "Write a concise summary of at most 3 sentences or a short bulleted list (≤5 bullets). "
                "Pack in specific numbers and facts; omit generic filler. Be brief.\n\n"
                "TEXT FRAGMENTS:\n"
                "{chunks_text}\n\n"
                "SUMMARY:"
            )

    async def _call_openrouter(self, prompt: str) -> str:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Железобетонная проверка типа ответа
                if isinstance(response, str):
                    return response.strip()
                
                # Если это стандартный объект OpenAI / AsyncOpenAI
                if hasattr(response, 'choices') and response.choices:
                    return response.choices[0].message.content.strip()
                    
                # На случай, если это словарь (dict)
                if isinstance(response, dict):
                    return response['choices'][0]['message']['content'].strip()

                return str(response).strip()

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    print(f"OpenRouter error: {e}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return ""
        return ""


    async def _process_window(self, window: List[Chunk], idx: int, filepath: str, pbar) -> Chunk:
        combined_text = "\n---\n".join(ch.text for ch in window)
        summary = await self._call_openrouter(
            self.prompt_template.format(chunks_text=combined_text)
        )
        pbar.update(1)
        
        if not summary:
            return None

        if self.include_source_refs:
            source_ids = [str(ch.pos) for ch in window]
            summary += f"\n\n[SourceChunks: {','.join(source_ids)}]"

        return Chunk(
            text=summary,
            doc=filepath,
            pos=idx,
            synth=True
        )

    def chunk(self, filepath: str = "") -> List[Chunk]:
        from threading import Thread

        original_chunks = self.base_chunker.chunk(filepath)
        all_chunks = list(original_chunks)

        windows = []
        for start in range(0, len(original_chunks), self.window_size):
            window = original_chunks[start:start + self.window_size]
            if len(window) >= 2:
                windows.append(window)

        if not windows:
            return all_chunks

        async def run_pipeline():
            sem = asyncio.Semaphore(self.max_concurrent)

            async def bounded(window, i, pbar):
                async with sem:
                    return await self._process_window(window, i, filepath, pbar)

            with tqdm.tqdm(total=len(windows), desc="Generating summaries", unit="window", leave=False) as pbar:
                tasks = [bounded(window, i, pbar) for i, window in enumerate(windows)]
                return await asyncio.gather(*tasks)

        outputs = {}

        def thread_target():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                outputs['result'] = new_loop.run_until_complete(run_pipeline())
            except Exception as e:
                outputs['exception'] = e
            finally:
                new_loop.close()

        # Изолированный поток защитит от конфликтов вложенных чанкеров в precalc()
        worker_thread = Thread(target=thread_target)
        worker_thread.start()
        worker_thread.join()

        if 'exception' in outputs:
            raise outputs['exception']

        synthetic_chunks = outputs.get('result', [])

        next_pos = len(all_chunks)
        for chunk in synthetic_chunks:
            if chunk is not None:
                chunk.pos = next_pos
                all_chunks.append(chunk)
                next_pos += 1

        return all_chunks
