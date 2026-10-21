"""Credential-free provider errors shared by HTTP adapters and the API."""


class EmbeddingUnavailable(ValueError):
    """Retryable provider failure with no provider body or URL in its message."""
