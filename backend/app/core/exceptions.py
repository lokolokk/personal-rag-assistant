"""Custom domain exceptions for the Conversational RAG platform."""

from typing import Any


class AppException(Exception):
    """Base application exception with standardized code and HTTP status code."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Any = None,
    ) -> None:
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class DocumentNotFoundError(AppException):
    """Raised when a requested document ID does not exist in registry or stores."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with ID '{document_id}' does not exist.",
            code="DOCUMENT_NOT_FOUND",
            status_code=404,
        )


class InvalidFileFormatError(AppException):
    """Raised when an uploaded file is not a supported PDF format."""

    def __init__(self, message: str = "Only PDF files are supported.") -> None:
        super().__init__(
            message=message,
            code="INVALID_FILE_TYPE",
            status_code=400,
        )


class FileSizeLimitExceededError(AppException):
    """Raised when an uploaded file exceeds the 30MB size limit."""

    def __init__(self, message: str = "File exceeds maximum size limit of 30MB.") -> None:
        super().__init__(
            message=message,
            code="FILE_TOO_LARGE",
            status_code=413,
        )


class DocumentParsingError(AppException):
    """Raised when parsing a document fails due to corruption or formatting errors."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            code="DOCUMENT_PARSING_ERROR",
            status_code=400,
        )


class DocumentIngestionError(AppException):
    """Raised when document chunking or indexing fails."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            code="INGESTION_ERROR",
            status_code=500,
        )


class LLMConfigurationError(AppException):
    """Raised when an LLM operation is invoked without required credentials or configuration."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            code="LLM_CONFIGURATION_ERROR",
            status_code=500,
        )


class QueryReformulationError(AppException):
    """Raised when multi-turn query reformulation fails during LLM invocation."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message=message,
            code="QUERY_REFORMULATION_ERROR",
            status_code=500,
        )


class ConversationNotFoundError(AppException):
    """Raised when a requested conversation ID does not exist in registry or memory."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            message=f"Conversation with ID '{conversation_id}' does not exist.",
            code="CONVERSATION_NOT_FOUND",
            status_code=404,
        )
