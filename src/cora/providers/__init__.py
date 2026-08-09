"""cora's provider seams — pluggable backends the engine talks to.

Each provider is an ABC with a sensible default implementation, so cora
runs standalone (no extra infrastructure) and adopters swap in their
own: retrieval, git/SCM introspection, and the reporting sink.
"""

from cora.providers.git import GitProvider, LocalGitProvider
from cora.providers.reporter import GitHubReporter, NullReporter, Reporter
from cora.providers.retrieval import (
    GlobRetrievalProvider,
    NullRetrievalProvider,
    RetrievalProvider,
    TeiQdrantRetrievalProvider,
)

__all__ = [
    "GitHubReporter",
    "GitProvider",
    "GlobRetrievalProvider",
    "LocalGitProvider",
    "NullReporter",
    "NullRetrievalProvider",
    "Reporter",
    "RetrievalProvider",
    "TeiQdrantRetrievalProvider",
]
