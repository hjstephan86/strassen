# test_matrix.py

import unittest
from src.matrix import Matrix # Make sure matrix.py is in the same directory or accessible

class TestMatrix(unittest.TestCase):

    # --- Test __init__ (Constructor) ---
    def test_init_valid_matrix(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.data, [[1, 2], [3, 4]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)

    def test_init_empty_matrix(self):
        m = Matrix([])
        self.assertEqual(m.data, [])
        self.assertEqual(m.rows, 0)
        self.assertEqual(m.cols, 0)

    def test_init_single_element_matrix(self):
        m = Matrix([[5]])
        self.assertEqual(m.data, [[5]])
        self.assertEqual(m.rows, 1)
        self.assertEqual(m.cols, 1)

    def test_init_matrix_with_float_values(self):
        m = Matrix([[1.0, 2.5], [3.3, 4.7]])
        self.assertEqual(m.data, [[1.0, 2.5], [3.3, 4.7]])
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)

    def test_init_type_error_not_list(self):
        with self.assertRaises(TypeError) as cm:
            Matrix("not a list")
        self.assertEqual(str(cm.exception), "Matrix data must be a list of lists.")

    def test_init_type_error_not_list_of_lists(self):
        with self.assertRaises(TypeError) as cm:
            Matrix([1, 2, 3])
        self.assertEqual(str(cm.exception), "Matrix data must be a list of lists.")

    def test_init_value_error_ragged_rows(self):
        with self.assertRaises(ValueError) as cm:
            Matrix([[1, 2], [3]])
        self.assertEqual(str(cm.exception), "All rows in the matrix must have the same length.")

    def test_init_deep_copy(self):
        original_data = [[1, 2], [3, 4]]
        m = Matrix(original_data)
        original_data[0][0] = 99 # Modify original data
        self.assertEqual(m.data, [[1, 2], [3, 4]]) # Matrix data should be unchanged

    # --- Test get_dimensions ---
    def test_get_dimensions(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.get_dimensions(), (2, 3))

    def test_get_dimensions_empty(self):
        m = Matrix([])
        self.assertEqual(m.get_dimensions(), (0, 0))

    # --- Test __getitem__ ---
    def test_getitem_valid_row(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(m[0], [1, 2])
        self.assertEqual(m[1], [3, 4])

    def test_getitem_type_error(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError) as cm:
            m["0"]
        self.assertEqual(str(cm.exception), "Row index must be an integer.")

    def test_getitem_index_error_negative(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(IndexError) as cm:
            m[-1]
        self.assertEqual(str(cm.exception), "Row index out of bounds.")

    def test_getitem_index_error_out_of_bounds(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(IndexError) as cm:
            m[2]
        self.assertEqual(str(cm.exception), "Row index out of bounds.")

    def test_getitem_on_empty_matrix(self):
        m = Matrix([])
        with self.assertRaises(IndexError):
            m[0]

    # --- Test __setitem__ ---
    def test_setitem_valid_row(self):
        m = Matrix([[1, 2], [3, 4]])
        m[0] = [5, 6]
        self.assertEqual(m.data, [[5, 6], [3, 4]])

    def test_setitem_type_error(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError) as cm:
            m["0"] = [7, 8]
        self.assertEqual(str(cm.exception), "Row index must be an integer.")

    def test_setitem_index_error_out_of_bounds(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(IndexError) as cm:
            m[2] = [7, 8]
        self.assertEqual(str(cm.exception), "Row index out of bounds.")

    def test_setitem_value_error_wrong_length(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(ValueError) as cm:
            m[0] = [7, 8, 9]
        self.assertEqual(str(cm.exception), "New row must be a list of length 2.")

    def test_setitem_value_error_not_list(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(ValueError) as cm:
            m[0] = "not a list"
        self.assertEqual(str(cm.exception), "New row must be a list of length 2.")

    def test_setitem_deep_copy(self):
        m = Matrix([[1, 2], [3, 4]])
        new_row = [5, 6]
        m[0] = new_row
        new_row[0] = 99 # Modify original list
        self.assertEqual(m.data, [[5, 6], [3, 4]]) # Matrix data should be unchanged

    # --- Test __str__ and __repr__ ---
    def test_str_representation(self):
        m = Matrix([[1, 2], [3, 4]])
        expected_str = "1\t2\n3\t4"
        self.assertEqual(str(m), expected_str)

    def test_str_representation_empty(self):
        m = Matrix([])
        self.assertEqual(str(m), "[]")

    def test_str_representation_single_row(self):
        m = Matrix([[10, 20, 30]])
        self.assertEqual(str(m), "10\t20\t30")

    def test_repr_representation(self):
        m = Matrix([[1, 2], [3, 4]])
        self.assertEqual(repr(m), "Matrix([[1, 2], [3, 4]])")

    def test_repr_representation_empty(self):
        m = Matrix([])
        self.assertEqual(repr(m), "Matrix([])")

    # --- Test __eq__ ---
    def test_eq_identical_matrices(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 4]])
        self.assertTrue(m1 == m2)

    def test_eq_different_data(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2], [3, 5]])
        self.assertFalse(m1 == m2)

    def test_eq_different_dimensions(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(m1 == m2)

    def test_eq_different_type(self):
        m1 = Matrix([[1, 2], [3, 4]])
        self.assertNotEqual(m1, "not a matrix") # Uses NotImplemented then falls back to default

    def test_eq_empty_matrices(self):
        m1 = Matrix([])
        m2 = Matrix([])
        self.assertTrue(m1 == m2)

    def test_eq_empty_and_non_empty(self):
        m1 = Matrix([])
        m2 = Matrix([[1]])
        self.assertFalse(m1 == m2)

    # --- Test __add__ ---
    def test_add_valid_matrices(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[6, 8], [10, 12]])
        self.assertEqual(m1 + m2, expected)

    def test_add_type_error(self):
        m1 = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError) as cm:
            m1 + 5
        self.assertEqual(str(cm.exception), "Can only add a Matrix object to a Matrix object.")

    def test_add_value_error_dimension_mismatch(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6, 7], [8, 9, 0]])
        with self.assertRaises(ValueError) as cm:
            m1 + m2
        self.assertEqual(str(cm.exception), "Matrices must have the same dimensions for addition.")

    def test_add_with_empty_matrices(self):
        m1 = Matrix([])
        m2 = Matrix([])
        expected = Matrix([])
        self.assertEqual(m1 + m2, expected)

    # --- Test __sub__ ---
    def test_sub_valid_matrices(self):
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])
        expected = Matrix([[4, 4], [4, 4]])
        self.assertEqual(m1 - m2, expected)

    def test_sub_type_error(self):
        m1 = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError) as cm:
            m1 - 5
        self.assertEqual(str(cm.exception), "Can only subtract a Matrix object from a Matrix object.")

    def test_sub_value_error_dimension_mismatch(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6, 7], [8, 9, 0]])
        with self.assertRaises(ValueError) as cm:
            m1 - m2
        self.assertEqual(str(cm.exception), "Matrices must have the same dimensions for subtraction.")

    def test_sub_with_empty_matrices(self):
        m1 = Matrix([])
        m2 = Matrix([])
        expected = Matrix([])
        self.assertEqual(m1 - m2, expected)

    # --- Test __mul__ (Multiplication) ---
    def test_mul_scalar_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        scalar = 2
        expected = Matrix([[2, 4], [6, 8]])
        self.assertEqual(m * scalar, expected)

    def test_mul_scalar_multiplication_by_zero(self):
        m = Matrix([[1, 2], [3, 4]])
        scalar = 0
        expected = Matrix([[0, 0], [0, 0]])
        self.assertEqual(m * scalar, expected)

    def test_mul_scalar_multiplication_with_float(self):
        m = Matrix([[1, 2], [3, 4]])
        scalar = 0.5
        expected = Matrix([[0.5, 1.0], [1.5, 2.0]])
        self.assertEqual(m * scalar, expected)

    def test_mul_matrix_multiplication_square(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        expected = Matrix([[19, 22], [43, 50]]) # (1*5+2*7, 1*6+2*8), (3*5+4*7, 3*6+4*8)
        self.assertEqual(m1 * m2, expected)

    def test_mul_matrix_multiplication_rectangular(self):
        m1 = Matrix([[1, 2, 3], [4, 5, 6]]) # 2x3
        m2 = Matrix([[7, 8], [9, 1], [2, 3]]) # 3x2
        expected = Matrix([[31, 19], [85, 55]])
        # (1*7+2*9+3*2 = 7+18+6 = 31), (1*8+2*1+3*3 = 8+2+9 = 19)
        # (4*7+5*9+6*2 = 28+45+12 = 85), (4*8+5*1+6*3 = 32+5+18 = 55)
        self.assertEqual(m1 * m2, expected)

    def test_mul_matrix_multiplication_identity(self):
        m1 = Matrix([[1, 0], [0, 1]]) # Identity
        m2 = Matrix([[5, 6], [7, 8]])
        self.assertEqual(m1 * m2, m2)

    def test_mul_matrix_multiplication_by_zero_matrix(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[0, 0], [0, 0]])
        expected = Matrix([[0, 0], [0, 0]])
        self.assertEqual(m1 * m2, expected)

    def test_mul_value_error_dimension_mismatch(self):
        m1 = Matrix([[1, 2], [3, 4]]) # 2x2
        m2 = Matrix([[5, 6, 7]]) # 1x3
        with self.assertRaises(ValueError) as cm:
            m1 * m2
        self.assertIn("Number of columns in the first matrix (2) must equal number of rows in the second matrix (1) for multiplication.", str(cm.exception))

    def test_mul_type_error_unsupported_operand(self):
        m = Matrix([[1, 2], [3, 4]])
        with self.assertRaises(TypeError) as cm:
            m * "invalid"
        self.assertIn("Unsupported operand type for multiplication.", str(cm.exception))

    # --- Test __rmul__ (Reverse Multiplication) ---
    def test_rmul_scalar_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        scalar = 2
        expected = Matrix([[2, 4], [6, 8]])
        self.assertEqual(scalar * m, expected)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)