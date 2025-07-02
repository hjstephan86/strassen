# strassen_test.py

import unittest
import math
from src.matrix import Matrix # Make sure matrix.py is in the same directory or accessible
from src.strassen import (
    _get_next_power_of_2,
    _split_matrix,
    _join_matrices,
    _strassen_recursive_helper,
    strassen_multiply
)

class TestStrassenAlgorithms(unittest.TestCase):

    # --- Test _get_next_power_of_2 ---
    def test_get_next_power_of_2_exact(self):
        self.assertEqual(_get_next_power_of_2(2), 2)
        self.assertEqual(_get_next_power_of_2(4), 4)
        self.assertEqual(_get_next_power_of_2(8), 8)
        self.assertEqual(_get_next_power_of_2(16), 16)

    def test_get_next_power_of_2_non_exact(self):
        self.assertEqual(_get_next_power_of_2(1), 1) # Smallest power of 2
        self.assertEqual(_get_next_power_of_2(3), 4)
        self.assertEqual(_get_next_power_of_2(5), 8)
        self.assertEqual(_get_next_power_of_2(7), 8)
        self.assertEqual(_get_next_power_of_2(9), 16)
        self.assertEqual(_get_next_power_of_2(15), 16)

    def test_get_next_power_of_2_zero(self):
        self.assertEqual(_get_next_power_of_2(0), 0)

    # --- Test _split_matrix ---
    def test_split_matrix_2x2(self):
        matrix = Matrix([[1, 2], [3, 4]])
        A11, A12, A21, A22 = _split_matrix(matrix)
        self.assertEqual(A11, Matrix([[1]]))
        self.assertEqual(A12, Matrix([[2]]))
        self.assertEqual(A21, Matrix([[3]]))
        self.assertEqual(A22, Matrix([[4]]))

    def test_split_matrix_4x4(self):
        matrix = Matrix([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ])
        A11, A12, A21, A22 = _split_matrix(matrix)
        self.assertEqual(A11, Matrix([[1, 2], [5, 6]]))
        self.assertEqual(A12, Matrix([[3, 4], [7, 8]]))
        self.assertEqual(A21, Matrix([[9, 10], [13, 14]]))
        self.assertEqual(A22, Matrix([[11, 12], [15, 16]]))

    def test_split_matrix_odd_dimension_raises_error(self):
        matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with self.assertRaises(ValueError) as cm:
            _split_matrix(matrix)
        self.assertEqual(str(cm.exception), "Matrix dimension must be even for splitting.")

    def test_split_matrix_1x1_raises_error(self):
        matrix = Matrix([[1]])
        with self.assertRaises(ValueError) as cm:
            _split_matrix(matrix)
        self.assertEqual(str(cm.exception), "Matrix dimension must be even for splitting.")

    # --- Test _join_matrices ---
    def test_join_matrices_1x1_to_2x2(self):
        C11 = Matrix([[1]])
        C12 = Matrix([[2]])
        C21 = Matrix([[3]])
        C22 = Matrix([[4]])
        joined_matrix = _join_matrices(C11, C12, C21, C22)
        self.assertEqual(joined_matrix, Matrix([[1, 2], [3, 4]]))

    def test_join_matrices_2x2_to_4x4(self):
        C11 = Matrix([[1, 2], [5, 6]])
        C12 = Matrix([[3, 4], [7, 8]])
        C21 = Matrix([[9, 10], [13, 14]])
        C22 = Matrix([[11, 12], [15, 16]])
        joined_matrix = _join_matrices(C11, C12, C21, C22)
        self.assertEqual(joined_matrix, Matrix([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]))

    # --- Test _strassen_recursive_helper ---
    def test_strassen_recursive_helper_base_case(self):
        # Using threshold=2, so 2x2 should use standard multiplication
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        expected = A * B # Standard multiplication
        result = _strassen_recursive_helper(A, B, threshold=2)
        self.assertEqual(result, expected)

        # Using threshold=1 for 1x1 matrices
        A_1x1 = Matrix([[5]])
        B_1x1 = Matrix([[10]])
        expected_1x1 = A_1x1 * B_1x1
        result_1x1 = _strassen_recursive_helper(A_1x1, B_1x1, threshold=1)
        self.assertEqual(result_1x1, expected_1x1)


    def test_strassen_recursive_helper_2x2_recursive(self):
        # Using threshold=1, forces 2x2 to go through recursive path
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[19, 22], [43, 50]]) # (1*5+2*7, 1*6+2*8), (3*5+4*7, 3*6+4*8)
        result = _strassen_recursive_helper(A, B, threshold=1)
        self.assertEqual(result, expected)

    def test_strassen_recursive_helper_4x4_recursive(self):
        # Identity matrix * another matrix should result in the same matrix
        A = Matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        B = Matrix([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ])
        expected = B
        result = _strassen_recursive_helper(A, B, threshold=1)
        self.assertEqual(result, expected)

    # --- Test strassen_multiply (Public API) ---

    def test_strassen_multiply_2x2(self):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[5, 6], [7, 8]])
        expected = A * B
        result = strassen_multiply(A, B, threshold=1) # Forces full Strassen
        self.assertEqual(result, expected)

    def test_strassen_multiply_4x4(self):
        A = Matrix([
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 3, 4]
        ])
        B = Matrix([
            [5, 6, 0, 0],
            [7, 8, 0, 0],
            [0, 0, 5, 6],
            [0, 0, 7, 8]
        ])
        expected = A * B
        result = strassen_multiply(A, B, threshold=1)
        self.assertEqual(result, expected)

    def test_strassen_multiply_rectangular_matrices_requiring_padding(self):
        A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2x4
        B = Matrix([[1, 0], [0, 1], [1, 0], [0, 1]]) # 4x2
        expected = A * B # 2x2
        result = strassen_multiply(A, B, threshold=1)
        self.assertEqual(result, expected)

    def test_strassen_multiply_odd_square_matrices_requiring_padding(self):
        A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 3x3
        B = Matrix([[9, 8, 7], [6, 5, 4], [3, 2, 1]]) # 3x3
        expected = A * B
        result = strassen_multiply(A, B) # Uses default threshold
        self.assertEqual(result, expected)

    def test_strassen_multiply_mixed_dimensions_requiring_padding(self):
        A = Matrix([[1, 2], [3, 4], [5, 6]]) # 3x2
        B = Matrix([[7, 8, 9, 10], [11, 12, 13, 14]]) # 2x4
        expected = A * B # 3x4
        result = strassen_multiply(A, B)
        self.assertEqual(result, expected)

    def test_strassen_multiply_with_zero_matrix(self):
        A = Matrix([[0, 0], [0, 0]])
        B = Matrix([[1, 2], [3, 4]])
        expected = A * B
        result = strassen_multiply(A, B, threshold=1)
        self.assertEqual(result, expected)

        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([[0, 0], [0, 0]])
        expected = A * B
        result = strassen_multiply(A, B, threshold=1)
        self.assertEqual(result, expected)

    def test_strassen_multiply_large_matrices_not_power_of_2(self):
        # 5x5 matrix, will be padded to 8x8
        A_data = [[i*5 + j + 1 for j in range(5)] for i in range(5)]
        B_data = [[(i+j)*2 for j in range(5)] for i in range(5)]
        A = Matrix(A_data)
        B = Matrix(B_data)
        expected = A * B
        result = strassen_multiply(A, B) # Default threshold
        self.assertEqual(result, expected)

    def test_strassen_multiply_1x1_matrices(self):
        A = Matrix([[5]])
        B = Matrix([[10]])
        expected = Matrix([[50]])
        result = strassen_multiply(A, B, threshold=1)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)