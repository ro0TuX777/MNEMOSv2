"""Non-sensitive structured errors for the isolated adapter."""

from __future__ import annotations


class ShadowAdapterError(Exception):
    def __init__(self, code: str, safe_message: str, retryable: bool = False):
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable

    def response(self, request_id: str | None, contract_version: str) -> dict:
        return {
            "request_id": request_id,
            "adapter_contract_version": contract_version,
            "error_code": self.code,
            "retryable": self.retryable,
            "safe_retry_after": None,
        }
