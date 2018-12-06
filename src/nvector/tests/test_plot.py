"""
Created on 6. des. 2018

@author: pab
"""
from __future__ import absolute_import
import sys
from ..plot import plot_mean_position
import pytest


@pytest.mark.xfail(sys.version_info == (3, 5),
                   reason="Does not work on python3.5")
def test_plot_mean_position():
    plot_mean_position()
