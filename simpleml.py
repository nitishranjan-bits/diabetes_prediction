# Module docstring explaining the purpose of the module
"""
This module contains functions related to number operations.
"""

# Global variable example
numbers = [10.5, 21.5, 35.5]

def calculate_sum():
    """
    Calculate the sum of numbers.
    """
    # Access the global 'numbers' list directly
    total_sum = sum(numbers)
    return total_sum

def calculate_product():
    """
    Calculate the product of numbers.
    """
    return numbers[0] * numbers[1] * numbers[2]

# Example usage
if __name__ == "__main__":
    print(f"Sum of numbers: {calculate_sum()}")
    print(f"Product of numbers: {calculate_product()}")
