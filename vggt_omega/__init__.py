# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bundled VGGT backbone kept for checkpoint compatibility with SelfTR."""

from .models import VGGTOmega

__version__ = "0.1.0"

__all__ = ["VGGTOmega", "__version__"]
