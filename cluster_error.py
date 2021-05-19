#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Error(Exception):
    """Base class for other exceptions"""

    pass


class OutOfBoundsClusterError(Error):
    """Raised when the Cluster is out of bounds of dimensions of the image"""

    pass


class ClusterNotGeneratedError(Error):
    """Raised when Add Cluster is called before Generate Cluster"""

    pass


class UndoError(Error):
    """Raised when Add Cluster is called before Generate Cluster"""

    pass
