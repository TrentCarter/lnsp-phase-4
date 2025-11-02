#!/usr/bin/env python3
"""
Model Sanity Test
=================

Tests if a model's forward pass produces expected results.

Loads a model and validation data, then:
1. Runs forward pass
2. Computes cosine with target
3. Checks if results match training metrics

This verifies there's no sign flip or normalization issue.
