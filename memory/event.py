"""Source-only event compatibility interfaces."""
from .typed_sources import SourceBuilder, SourceRetriever


class EventBuilder(SourceBuilder):
    def __init__(self, db_path):
        super().__init__(db_path, 'event')


class EventRetriever(SourceRetriever):
    def __init__(self, db_path, embedder=None):
        super().__init__(db_path, 'event', embedder)
