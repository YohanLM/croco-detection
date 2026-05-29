"""Efficiency metrics for CRC - per-image prediction-set size."""

from conformal.efficiency.box_area import total_box_area
from conformal.efficiency.box_count import box_count

__all__ = ["total_box_area", "box_count"]
