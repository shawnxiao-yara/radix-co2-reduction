"""Test package version."""

from radix_co2_reduction import __version__


def test_version() -> None:
    """Test that the version string can be loaded."""
    assert isinstance(__version__, str)
