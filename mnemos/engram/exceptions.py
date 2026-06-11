class ArtifactPolicyRejectedError(Exception):
    """Raised when an ingestion artifact explicitly violates memory policy."""
    pass
