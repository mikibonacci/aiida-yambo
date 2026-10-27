# -*- coding: utf-8 -*-
"""Helpers of the YamboConvergence workflow."""
from __future__ import absolute_import

###############################  parallelism  ####################################

def build_parallelism_instructions(instructions):
    """Normalize parallelism instruction keys for workflow execution."""

    instructions['automatic'] = instructions.pop('automatic', False)
    instructions['semi-automatic'] = instructions.pop('semi-automatic', False)
    instructions['manual'] = instructions.pop('manual', False)
    instructions['function'] = instructions.pop('function', False)

    return instructions
