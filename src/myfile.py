"""
This module contains utility functions for various tasks.
"""

from typing import List, Optional

def add_numbers(a: int, b: int) -> int:
    """
    Adds two numbers and returns the result.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The sum of the two numbers.
    """
    return a + b

def find_max(numbers: List[int]) -> Optional[int]:
    """
    Finds the maximum number in a list.

    Args:
        numbers (List[int]): A list of integers.

    Returns:
        Optional[int]: The maximum number in the list, or None if the list is empty.
    """
    if not numbers:
        return None
    return max(numbers)
