"""Exact finite-size RBM and three-body RBM experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ip-rbm")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
