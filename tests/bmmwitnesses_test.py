# bmmwitnesses_test.py

import unittest
from src.matrix import Matrix # Assuming your Matrix class
from src.strassen import strassen_multiply # Assuming your strassen_multiply function
from src.bmmwitnesses import boolean_matrix_multiply, bmm_witness

class TestBMMWitnesses(unittest.TestCase):

    def setUp(self):
        # Sample matrices for testing
        self.A_small = Matrix([[1, 0], [0, 1]]) # Identity
        self.B_small = Matrix([[0, 1], [1, 0]]) # Permutation

        self.A_3x3 = Matrix([[1, 0, 1],
                             [0, 1, 0],
                             [1, 1, 0]])
        self.B_3x3 = Matrix([[0, 1, 0],
                             [1, 0, 1],
                             [0, 1, 1]])
        
        self.A_2x3 = Matrix([[1, 0, 1],
                             [0, 1, 0]])
        self.B_3x2 = Matrix([[0, 1],
                             [1, 0],
                             [0, 1]])

        # Matrices for specific witness tests
        self.A_no_witness = Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.B_no_witness = Matrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

        self.A_all_witness = Matrix([[1, 1], [1, 1]])
        self.B_all_witness = Matrix([[1, 1], [1, 1]])


    # --- Tests for boolean_matrix_multiply ---

    def test_bmm_naive_small_matrices(self):
        """Test boolean_matrix_multiply with naive approach on small matrices."""
        expected_C = Matrix([[0, 1], [1, 0]])
        result_C = boolean_matrix_multiply(self.A_small, self.B_small, use_strassen=False)
        self.assertEqual(result_C, expected_C)

    def test_bmm_naive_3x3_matrices(self):
        """Test boolean_matrix_multiply with naive approach on 3x3 matrices."""
        expected_C = Matrix([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]])
        result_C = boolean_matrix_multiply(self.A_3x3, self.B_3x3, use_strassen=False)
        self.assertEqual(result_C, expected_C)

    def test_bmm_strassen_small_matrices(self):
        """Test boolean_matrix_multiply with Strassen approach on small matrices."""
        expected_C = Matrix([[0, 1], [1, 0]])
        result_C = boolean_matrix_multiply(self.A_small, self.B_small, use_strassen=True, strassen_threshold=1)
        self.assertEqual(result_C, expected_C)

    def test_bmm_strassen_3x3_matrices(self):
        """Test boolean_matrix_multiply with Strassen approach on 3x3 matrices."""
        expected_C = Matrix([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]])
        result_C = boolean_matrix_multiply(self.A_3x3, self.B_3x3, use_strassen=True, strassen_threshold=1)
        self.assertEqual(result_C, expected_C)

    def test_bmm_strassen_matches_naive(self):
        """Ensure Strassen BMM gives the same result as naive BMM."""
        large_N = 64 # Choose a size where Strassen might be faster
        A_large = Matrix([[1 if (i + j) % 3 == 0 else 0 for j in range(large_N)] for i in range(large_N)])
        B_large = Matrix([[1 if (i * j) % 4 == 0 else 0 for j in range(large_N)] for i in range(large_N)])

        result_naive = boolean_matrix_multiply(A_large, B_large, use_strassen=False)
        result_strassen = boolean_matrix_multiply(A_large, B_large, use_strassen=True, strassen_threshold=16)
        self.assertEqual(result_naive, result_strassen, "Strassen BMM result must match Naive BMM result.")

    def test_bmm_incompatible_dimensions(self):
        """Test boolean_matrix_multiply with incompatible dimensions."""
        A = Matrix([[1, 0]]) # 1x2
        B = Matrix([[1, 0]]) # 1x2
        with self.assertRaises(ValueError):
            boolean_matrix_multiply(A, B)

    def test_bmm_non_matrix_input(self):
        """Test boolean_matrix_multiply with non-Matrix inputs."""
        A = Matrix([[1, 0]])
        with self.assertRaises(TypeError):
            boolean_matrix_multiply(A, [[0, 1]])
        with self.assertRaises(TypeError):
            boolean_matrix_multiply([[1, 0]], A)

    def test_bmm_rectangular_matrices(self):
        """Test boolean_matrix_multiply with compatible rectangular matrices."""
        expected_C = Matrix([[0, 1], [1, 0]])
        result_C = boolean_matrix_multiply(self.A_2x3, self.B_3x2, use_strassen=False)
        self.assertEqual(result_C, expected_C)
        
        result_C_strassen = boolean_matrix_multiply(self.A_2x3, self.B_3x2, use_strassen=True, strassen_threshold=1)
        self.assertEqual(result_C_strassen, expected_C)


    # --- Tests for bmm_witness ---

    def test_bmm_witness_basic_small(self):
        """Test bmm_witness on small matrices."""
        expected_C = Matrix([[0, 1], [1, 0]])
        expected_W = Matrix([[0, 1], [2, 0]]) 
        
        C, W = bmm_witness(self.A_small, self.B_small)
        self.assertEqual(C, expected_C)
        self.assertEqual(W, expected_W)

    def test_bmm_witness_3x3(self):
        """Test bmm_witness on 3x3 matrices."""
        expected_C = Matrix([[0, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]])
        # Expected W:
        # C[0][1]=1: A[0][0]&B[0][1] (1&1) -> k=0 (W[0][1]=1) (first found)
        # C[0][2]=1: A[0][2]&B[2][2] (1&1) -> k=2 (W[0][2]=3)
        # C[1][0]=1: A[1][1]&B[1][0] (1&1) -> k=1 (W[1][0]=2)
        # C[1][2]=1: A[1][1]&B[1][2] (1&1) -> k=1 (W[1][2]=2)
        # C[2][0]=1: A[2][1]&B[1][0] (1&1) -> k=1 (W[2][0]=2)
        # C[2][1]=1: A[2][0]&B[0][1] (1&1) -> k=0 (W[2][1]=1)
        # C[2][2]=1: A[2][1]&B[1][2] (1&1) -> k=1 (W[2][2]=2)
        expected_W = Matrix([[0, 1, 3],
                             [2, 0, 2],
                             [2, 1, 2]])
        
        C, W = bmm_witness(self.A_3x3, self.B_3x3)
        self.assertEqual(C, expected_C)
        self.assertEqual(W, expected_W)

    def test_bmm_witness_no_path(self):
        """Test bmm_witness when no path exists."""
        expected_C = Matrix([[0, 0, 1],
                             [0, 0, 0],
                             [1, 0, 0]])
        expected_W = Matrix([[0, 0, 1], # A[0][0]&B[0][2] (k=0 for 0,2)
                             [0, 0, 0],
                             [3, 0, 0]]) # A[2][2]&B[2][0] (k=2 for 2,0)
        
        C, W = bmm_witness(self.A_no_witness, self.B_no_witness)
        self.assertEqual(C, expected_C)
        self.assertEqual(W, expected_W)

    def test_bmm_witness_all_path(self):
        """Test bmm_witness when all paths exist."""
        expected_C = Matrix([[1, 1], [1, 1]])
        expected_W = Matrix([[1, 1], [1, 1]]) # First k found will be 0 for all
        
        C, W = bmm_witness(self.A_all_witness, self.B_all_witness)
        self.assertEqual(C, expected_C)
        self.assertEqual(W, expected_W)

    def test_bmm_witness_incompatible_dimensions(self):
        """Test bmm_witness with incompatible dimensions."""
        A = Matrix([[1, 0]]) # 1x2
        B = Matrix([[1, 0]]) # 1x2
        with self.assertRaises(ValueError):
            bmm_witness(A, B)

    def test_bmm_witness_non_matrix_input(self):
        """Test bmm_witness with non-Matrix inputs."""
        A = Matrix([[1, 0]])
        with self.assertRaises(TypeError):
            bmm_witness(A, [[0, 1]])
        with self.assertRaises(TypeError):
            bmm_witness([[1, 0]], A)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)