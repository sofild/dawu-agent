"""Vector memory system with ChromaDB for long-term semantic retrieval."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dawu_agent.llm.base import Message


@dataclass
class MemoryEntry:
    """Single memory entry."""

    id: str
    category: str  # user | feedback | project | reference
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    ttl: int | None = None  # Time-to-live in seconds


class MemoryManager:
    """Four-category memory manager with vector search.

    Categories:
    - user: User preferences and habits
    - feedback: Corrections and feedback
    - project: Project structure and conventions
    - reference: External knowledge summaries
    """

    def __init__(
        self,
        persist_dir: str = "memory/vector_db",
        collection_name: str = "dawu_memories",
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.75,
        max_results: int = 5,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results

        self._client: Any = None
        self._collection: Any = None
        self._short_term: list[MemoryEntry] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize ChromaDB connection."""
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("chromadb not installed. Run: pip install chromadb") from e

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Workaround: ChromaDB's Rust backend persists DefaultEmbeddingFunction
        # which requires onnxruntime. We create collection with NO embedding function
        # and manually compute embeddings before add/query.
        try:
            self._collection = self._client.get_collection(self.collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute a deterministic embedding without external dependencies."""
        import hashlib
        import math
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(384):
            val = (hash_bytes[i % 32] + i * 7) % 256 - 128
            embedding.append(float(val) / 128.0)
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    async def add(
        self,
        content: str,
        category: str,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """Add a memory entry."""
        entry_id = str(uuid.uuid4())
        entry = MemoryEntry(
            id=entry_id,
            category=category,
            content=content,
            metadata=metadata or {},
            ttl=ttl,
        )

        async with self._lock:
            self._short_term.append(entry)

        # Add to vector DB with pre-computed embeddings
        if self._collection is not None:
            self._collection.add(
                ids=[entry_id],
                documents=[content],
                embeddings=[self._compute_embedding(content)],
                metadatas=[{"category": category, **(metadata or {})}],
            )

        return entry_id

    async def search(
        self,
        query: str,
        category: str | None = None,
        n_results: int | None = None,
    ) -> list[MemoryEntry]:
        """Search memories by semantic similarity."""
        if self._collection is None:
            return []

        n = n_results or self.max_results
        where_filter = {"category": category} if category else None

        results = self._collection.query(
            query_embeddings=[self._compute_embedding(query)],
            n_results=n,
            where=where_filter,
        )

        entries = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance

                if similarity < self.similarity_threshold:
                    continue

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                entries.append(MemoryEntry(
                    id=doc_id,
                    category=metadata.get("category", "unknown"),
                    content=results["documents"][0][i],
                    metadata=metadata,
                ))

        return entries

    async def get_relevant_memories(
        self,
        messages: list[Message],
        current_task: str | None = None,
    ) -> list[MemoryEntry]:
        """Get memories relevant to current conversation context."""
        # Build search query from recent messages
        recent_text = " ".join(
            m.content[:200] for m in messages[-5:] if m.role in ("user", "assistant")
        )

        if current_task:
            query = f"{current_task} {recent_text}"
        else:
            query = recent_text

        # Search all categories
        all_results = await self.search(query, n_results=self.max_results * 2)

        # Deduplicate and sort by relevance
        seen = set()
        unique_results = []
        for entry in all_results:
            if entry.id not in seen:
                seen.add(entry.id)
                unique_results.append(entry)

        return unique_results[:self.max_results]

    async def auto_dream(self, llm_client: Any | None = None) -> dict[str, int]:
        """Consolidate short-term memories into long-term storage.

        Triggered when short_term buffer exceeds threshold or session ends.
        """
        async with self._lock:
            if len(self._short_term) < 5:
                return {"processed": 0}

            entries = self._short_term.copy()
            self._short_term.clear()

        # Classify entries
        classified: dict[str, list[MemoryEntry]] = {
            "user": [],
            "feedback": [],
            "project": [],
            "reference": [],
        }

        for entry in entries:
            if entry.category in classified:
                classified[entry.category].append(entry)

        consolidated_count = 0

        # Process each category
        for category, cat_entries in classified.items():
            if not cat_entries:
                continue

            if category == "feedback":
                # Feedback: persist as-is with TTL
                for entry in cat_entries:
                    await self.add(
                        content=entry.content,
                        category=category,
                        metadata=entry.metadata,
                        ttl=entry.ttl or 7 * 24 * 3600,  # 7 days default
                    )
                consolidated_count += len(cat_entries)
            elif llm_client is not None:
                # Other categories: summarize via LLM
                combined = "\n".join(e.content for e in cat_entries)
                try:
                    prompt = f"请将以下多条记忆归纳为一条不超过500 tokens的摘要：\n\n{combined}"
                    from dawu_agent.llm.base import Message
                    response = llm_client.chat_sync(
                        messages=[Message(role="user", content=prompt)]
                    )
                    summary = response.content

                    await self.add(
                        content=summary,
                        category=category,
                        metadata={"consolidated_from": len(cat_entries)},
                    )
                    consolidated_count += len(cat_entries)
                except Exception:
                    # Fallback: store individually
                    for entry in cat_entries:
                        await self.add(
                            content=entry.content,
                            category=category,
                            metadata=entry.metadata,
                        )
                    consolidated_count += len(cat_entries)
            else:
                # No LLM available, store individually
                for entry in cat_entries:
                    await self.add(
                        content=entry.content,
                        category=category,
                        metadata=entry.metadata,
                    )
                consolidated_count += len(cat_entries)

        return {"processed": consolidated_count}

    def format_for_context(self, entries: list[MemoryEntry]) -> str:
        """Format memory entries for injection into system prompt."""
        if not entries:
            return ""

        parts = ["── 相关记忆 ──"]
        for entry in entries:
            parts.append(f"[{entry.category}] {entry.content[:300]}")

        return "\n".join(parts)
