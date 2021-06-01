"""Session support for the Earth Engine."""
import ee


def start() -> None:
    """Start a new session."""
    # Authenticate
    ee.Authenticate()

    # Initialize the lib
    ee.Initialize()
