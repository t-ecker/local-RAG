class LocalRagError(Exception):
    """base class for all expected errors"""


class ProviderError(LocalRagError):
    """Model provider failed"""


class ProviderUnavailableError(ProviderError):
    """Model provider isn't reachable"""


class ModelNotFoundError(ProviderError):
    """Requested model doesn't exist at the provider"""


class NoDocumentsError(LocalRagError):
    """Vector Store is empty"""


class NoSelectionError(LocalRagError):
    """No Documents or Mode selected to search in / with"""


class DocumentLoadError(LocalRagError):
    """Document couldn't be loaded / embedded"""


class NoExtractableTextError(LocalRagError):
    """Nothing could be extracted out of a document"""


class InvalidRetrievalModeError(LocalRagError):
    """Retrieval mode is unknown"""
