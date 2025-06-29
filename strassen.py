# strassen.py
import math
from matrix import Matrix # Import the Matrix class from the matrix module

def _get_next_power_of_2(n):
    """Helper to find the next power of 2 greater than or equal to n."""
    if n == 0:
        return 0 # Or handle as error/special case if 0 dimension is not expected
    return 2**math.ceil(math.log2(n))

def _split_matrix(matrix: Matrix):
    """
    Splits a square matrix into four sub-matrices: A11, A12, A21, A22.
    Assumes matrix dimension n is even.
    """
    n = matrix.get_dimensions()[0]
    if n % 2 != 0:
        raise ValueError("Matrix dimension must be even for splitting.")

    half_n = n // 2

    A11_data = [[0 for _ in range(half_n)] for _ in range(half_n)]
    A12_data = [[0 for _ in range(half_n)] for _ in range(half_n)]
    A21_data = [[0 for _ in range(half_n)] for _ in range(half_n)]
    A22_data = [[0 for _ in range(half_n)] for _ in range(half_n)]

    for r in range(n):
        for c in range(n):
            if r < half_n and c < half_n:
                A11_data[r][c] = matrix.data[r][c]
            elif r < half_n and c >= half_n:
                A12_data[r][c - half_n] = matrix.data[r][c]
            elif r >= half_n and c < half_n:
                A21_data[r - half_n][c] = matrix.data[r][c]
            else: # r >= half_n and c >= half_n
                A22_data[r - half_n][c - half_n] = matrix.data[r][c]

    return Matrix(A11_data), Matrix(A12_data), Matrix(A21_data), Matrix(A22_data)

def _join_matrices(C11: Matrix, C12: Matrix, C21: Matrix, C22: Matrix):
    """
    Joins four sub-matrices into a single larger matrix.
    Assumes all sub-matrices have the same dimension.
    """
    half_n = C11.get_dimensions()[0]
    n = half_n * 2

    C_data = [[0 for _ in range(n)] for _ in range(n)]

    for r in range(n):
        for c in range(n):
            if r < half_n and c < half_n:
                C_data[r][c] = C11.data[r][c]
            elif r < half_n and c >= half_n:
                C_data[r][c] = C12.data[r][c - half_n]
            elif r >= half_n and c < half_n:
                C_data[r][c] = C21.data[r - half_n][c]
            else:
                C_data[r][c] = C22.data[r - half_n][c - half_n]

    return Matrix(C_data)


def _strassen_recursive_helper(A: Matrix, B: Matrix, threshold: int):
    """
    Internal recursive helper for Strassen's algorithm.
    Assumes A and B are square matrices of dimension n, where n is a power of 2.
    """
    n = A.get_dimensions()[0]

    # Base case: if matrix is small enough, use standard multiplication
    if n <= threshold:
        return A * B # This uses the standard __mul__ from our Matrix class

    # Divide matrices into 4 sub-matrices
    A11, A12, A21, A22 = _split_matrix(A)
    B11, B12, B21, B22 = _split_matrix(B)

    # Calculate the 7 products recursively
    P1 = _strassen_recursive_helper(A11 + A22, B11 + B22, threshold)
    P2 = _strassen_recursive_helper(A21 + A22, B11, threshold)
    P3 = _strassen_recursive_helper(A11, B12 - B22, threshold)
    P4 = _strassen_recursive_helper(A22, B21 - B11, threshold)
    P5 = _strassen_recursive_helper(A11 + A12, B22, threshold)
    P6 = _strassen_recursive_helper(A21 - A11, B11 + B12, threshold)
    P7 = _strassen_recursive_helper(A12 - A22, B21 + B22, threshold)

    # Calculate the 4 result sub-matrices
    C11 = P1 + P4 - P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 - P2 + P3 + P6

    # Join the sub-matrices to form the final result
    return _join_matrices(C11, C12, C21, C22)


def strassen_multiply(A: Matrix, B: Matrix, threshold=16):
    """
    Implements Strassen's algorithm for matrix multiplication.
    Recursively multiplies two matrices using 7 multiplications.
    Pads matrices to powers of 2 if necessary, then unpads the result.

    Args:
        A (Matrix): The first matrix (rows_A x cols_A).
        B (Matrix): The second matrix (rows_B x cols_B).
        threshold (int): The dimension below which standard matrix
                         multiplication is used as the base case.
                         Typically between 16 and 128 for optimal performance.
    Returns:
        Matrix: The resulting product matrix (rows_A x cols_B).
    """
    rows_A, cols_A = A.get_dimensions()
    rows_B, cols_B = B.get_dimensions()

    if cols_A != rows_B:
        raise ValueError(
            f"Cannot multiply matrices: A's columns ({cols_A}) must match B's rows ({rows_B})"
        )

    # Determine the common padded dimension for square matrices
    # It must be large enough to contain both A and B's relevant dimensions for multiplication
    # and be a power of 2.
    max_dim = max(rows_A, cols_A, rows_B, cols_B)
    padded_dim = _get_next_power_of_2(max_dim)
    
    if padded_dim == 0 and (rows_A > 0 or cols_B > 0): # Handles edge case for empty/single element matrices
        padded_dim = 1 # Smallest valid power of 2 for Strassen if dimensions are minimal non-zero

    # Pad A to (padded_dim x padded_dim)
    A_padded_data = [[0 for _ in range(padded_dim)] for _ in range(padded_dim)]
    for r in range(rows_A):
        for c in range(cols_A):
            A_padded_data[r][c] = A.data[r][c]
    A_strassen = Matrix(A_padded_data)

    # Pad B to (padded_dim x padded_dim)
    B_padded_data = [[0 for _ in range(padded_dim)] for _ in range(padded_dim)]
    for r in range(rows_B):
        for c in range(cols_B):
            B_padded_data[r][c] = B.data[r][c]
    B_strassen = Matrix(B_padded_data)

    # Perform the recursive Strassen multiplication
    result_padded = _strassen_recursive_helper(A_strassen, B_strassen, threshold)

    # Unpad the result to the original dimensions (rows_A x cols_B)
    final_result_data = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for r in range(rows_A):
        for c in range(cols_B):
            final_result_data[r][c] = result_padded.data[r][c]

    return Matrix(final_result_data)


# --- Example Usage and Tests (when strassen.py is run directly) ---
if __name__ == "__main__":
    print("--- Running Strassen Algorithm Tests ---")

    # Test Case 1: Simple 2x2 matrices
    A1 = Matrix([[1, 2], [3, 4]])
    B1 = Matrix([[5, 6], [7, 8]])
    print(f"\nMatrix A1:\n{A1}")
    print(f"Matrix B1:\n{B1}")

    C_standard1 = A1 * B1
    print(f"Standard multiplication (A1 * B1):\n{C_standard1}")

    C_strassen1 = strassen_multiply(A1, B1, threshold=1)
    print(f"Strassen multiplication (A1 * B1):\n{C_strassen1}")
    assert C_standard1 == C_strassen1
    print("Test Case 1 Passed: Strassen result matches standard multiplication.")


    # Test Case 2: 4x4 matrices
    A2_data = [[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]] # Identity matrix
    B2_data = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
    A2 = Matrix(A2_data)
    B2 = Matrix(B2_data)
    print(f"\nMatrix A2:\n{A2}")
    print(f"Matrix B2:\n{B2}")

    C_standard2 = A2 * B2
    print(f"Standard multiplication (A2 * B2):\n{C_standard2}")

    C_strassen2 = strassen_multiply(A2, B2, threshold=1)
    print(f"Strassen multiplication (A2 * B2):\n{C_strassen2}")
    assert C_standard2 == C_strassen2
    print("Test Case 2 Passed: Strassen result matches standard multiplication.")


    # Test Case 3: Matrices that need padding (e.g., 3x3)
    A3 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B3 = Matrix([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    print(f"\nMatrix A3:\n{A3}")
    print(f"Matrix B3:\n{B3}")

    C_standard3 = A3 * B3
    print(f"Standard multiplication (A3 * B3):\n{C_standard3}")

    C_strassen3 = strassen_multiply(A3, B3)
    print(f"Strassen multiplication (A3 * B3):\n{C_strassen3}")
    assert C_standard3 == C_strassen3
    print("Test Case 3 Passed: Strassen result matches standard multiplication (with padding).")

    # Test Case 4: Rectangular matrices (will be padded to square then unpadded)
    A4 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2x4
    B4 = Matrix([[1, 0], [0, 1], [1, 0], [0, 1]]) # 4x2
    print(f"\nMatrix A4:\n{A4}")
    print(f"Matrix B4:\n{B4}")

    C_standard4 = A4 * B4
    print(f"Standard multiplication (A4 * B4):\n{C_standard4}")

    C_strassen4 = strassen_multiply(A4, B4, threshold=1)
    print(f"Strassen multiplication (A4 * B4):\n{C_strassen4}")
    assert C_standard4 == C_strassen4
    print("Test Case 4 Passed: Strassen result matches standard multiplication (rectangular).")

    # Test Case 5: Larger square matrix, not power of 2
    A5 = Matrix([
        [1,2,3,4,5],
        [6,7,8,9,10],
        [11,12,13,14,15],
        [16,17,18,19,20],
        [21,22,23,24,25]
    ])
    B5 = Matrix([
        [25,24,23,22,21],
        [20,19,18,17,16],
        [15,14,13,12,11],
        [10,9,8,7,6],
        [5,4,3,2,1]
    ])
    print(f"\nMatrix A5 (5x5):\n{A5}")
    print(f"Matrix B5 (5x5):\n{B5}")

    C_standard5 = A5 * B5
    print(f"Standard multiplication (A5 * B5):\n{C_standard5}")

    C_strassen5 = strassen_multiply(A5, B5)
    print(f"Strassen multiplication (A5 * B5):\n{C_strassen5}")
    assert C_standard5 == C_strassen5
    print("Test Case 5 Passed: Strassen result matches standard multiplication (5x5, padding to 8x8).")