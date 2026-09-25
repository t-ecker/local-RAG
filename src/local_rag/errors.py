class LocalRagError(Exception):
    """base class for all expected errors"""


class OllamaUnavailableError(LocalRagError):
    """Ollama isnt available"""


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
