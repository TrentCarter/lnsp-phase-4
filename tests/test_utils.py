#!/usr/bin/env python3
"""
Tests for utility functions in utils.py
"""

import unittest
from utils import hello, existing_function

class TestUtils(unittest.TestCase):
    def test_hello(self):
        """Test the hello function."""
        self.assertEqual(hello(), "Hello, World!")

    def test_existing_function(self):
        """Test the existing_function."""
        self.assertEqual(existing_function(), "I already exist")

if __name__ == '__main__':
    unittest.main()
