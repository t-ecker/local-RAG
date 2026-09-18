class LocalRagError(Exception):
    """base class for all expected errors"""


class OllamaUnavailableError(LocalRagError):
    """Ollama isnt available"""


class NoDocumentsError(LocalRagError):
    """Vector Store is empty"""


class NoSelectionError(LocalRagError):
    """No Docuemnts are selected to search in"""


class DocumentLoadError(LocalRagError):
    """Document couldnt be loaded / embedded"""


class NoExtractableTextError(LocalRagError):
    """Nothing could be extracted out of a document"""
