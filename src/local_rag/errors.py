class LocalRagError(Exception):
    """base class for all expected errors"""


class ProviderError(LocalRagError):
    """Model provider failed"""


class ProviderUnavailableError(ProviderError):
    """Model provider isnt reachable"""


class ModelNotFoundError(ProviderError):
    """Requested model doesnt exist at the provider"""


class NoDocumentsError(LocalRagError):
    """Vector Store is empty"""


class NoSelectionError(LocalRagError):
    """No Docuemnts or Mode selected to search in / with"""


class DocumentLoadError(LocalRagError):
    """Document couldnt be loaded / embedded"""


class NoExtractableTextError(LocalRagError):
    """Nothing could be extracted out of a document"""


class InvalidRetrieveModeError(LocalRagError):
    """Retrieve Mode is unknown"""
