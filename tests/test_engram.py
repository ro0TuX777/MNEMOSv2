"""Tests for the Engram model."""

import numpy as np
import pytest

from mnemos.engram.model import Engram, EngramBatch


class TestEngram:
    def test_default_creation(self):
        e = Engram(content="Hello world")
        assert e.content == "Hello world"
        assert e.confidence == 1.0
        assert e.neuro_tags == []
        assert e.edges == []
        assert e.id  # UUID should be generated

    def test_to_dict_no_embedding(self):
        e = Engram(id="test-1", content="hello", source="test")
        d = e.to_dict()
        assert d["id"] == "test-1"
        assert d["content"] == "hello"
        assert "embedding" not in d

    def test_to_dict_with_embedding(self):
        e = Engram(id="test-2", content="hello", embedding=np.ones(3, dtype=np.float32))
        d = e.to_dict(include_embedding=True)
        assert d["embedding"] == [1.0, 1.0, 1.0]

    def test_from_dict(self):
        data = {"id": "abc", "content": "doc", "source": "file.txt", "neuro_tags": ["science"]}
        e = Engram.from_dict(data)
        assert e.id == "abc"
        assert e.neuro_tags == ["science"]

    def test_add_tag_dedup(self):
        e = Engram()
        e.add_tag("physics")
        e.add_tag("physics")
        assert e.neuro_tags == ["physics"]

    def test_link(self):
        e = Engram()
        e.link("other-1")
        e.link("other-1")
        assert e.edges == ["other-1"]

    def test_round_trip(self):
        e = Engram(
            content="test",
            neuro_tags=["a", "b"],
            source="s3://bucket/file.pdf",
            confidence=0.95,
            metadata={"key": "value"},
            edges=["e1", "e2"],
        )
        d = e.to_dict()
        e2 = Engram.from_dict(d)
        assert e2.content == e.content
        assert e2.neuro_tags == e.neuro_tags
        assert e2.confidence == e.confidence


class TestEngramBatch:
    def test_batch_ops(self):
        batch = EngramBatch()
        batch.add(Engram(content="a"))
        batch.add(Engram(content="b"))
        assert len(batch) == 2
        assert batch.contents == ["a", "b"]

    def test_embeddings_stacking(self):
        batch = EngramBatch()
        batch.add(Engram(content="a", embedding=np.ones(4, dtype=np.float32)))
        batch.add(Engram(content="b", embedding=np.zeros(4, dtype=np.float32)))
        embs = batch.embeddings
        assert embs.shape == (2, 4)

    def test_empty_embeddings(self):
        batch = EngramBatch()
        batch.add(Engram(content="no embedding"))
        assert batch.embeddings is None
