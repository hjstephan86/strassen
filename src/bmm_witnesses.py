# bmm_witnesses.py

import time

from .matrix import Matrix # Your Matrix class
from .strassen import strassen_multiply # Your Strassen multiplication module

def boolean_matrix_multiply(A, B, use_strassen=False, strassen_threshold=32):
    """
    Performs Boolean Matrix Multiplication (A AND B) where C_ij = OR_k (A_ik AND B_kj).
    Assumes input matrices A and B contain only 0s and 1s.
    Returns a new Matrix object.
    
    Args:
        A (Matrix): The first matrix.
        B (Matrix): The second matrix.
        use_strassen (bool): If True, uses the external strassen_multiply for the
                             underlying arithmetic multiplication.
        strassen_threshold (int): Threshold for Strassen's algorithm (passed to strassen_multiply).
    """
    if not isinstance(A, Matrix) or not isinstance(B, Matrix):
        raise TypeError("Inputs must be Matrix objects.")

    rows_A, cols_A = A.get_dimensions()
    rows_B, cols_B = B.get_dimensions()

    if cols_A != rows_B:
        raise ValueError(
            f"Number of columns in A ({cols_A}) must equal "
            f"number of rows in B ({rows_B}) for Boolean matrix multiplication."
        )

    if use_strassen:
        # Perform standard arithmetic multiplication using your strassen_multiply
        # and then convert the result to Boolean.
        # This relies on the property that for 0/1 matrices, sum(A_ik * B_kj) > 0
        # is equivalent to OR_k(A_ik AND B_kj) = 1.
        arithmetic_product = strassen_multiply(A, B, threshold=strassen_threshold)
        
        result_data = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                if arithmetic_product[i][j] > 0:
                    result_data[i][j] = 1
        return Matrix(result_data)
    else:
        # Original standard Boolean multiplication
        result_data = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    if A[i][k] == 1 and B[k][j] == 1:
                        result_data[i][j] = 1
                        break
        return Matrix(result_data)

def bmm_witness(A, B):
    """
    Computes the Boolean Matrix Multiplication product C = A * B and
    a corresponding Witness Matrix W.
    W_ij will store a 'k' such that A_ik = 1 and B_kj = 1, if C_ij = 1.
    If C_ij = 0, W_ij will be 0 (or a similar indicator).

    Assumes input matrices A and B contain only 0s and 1s.
    Returns a tuple: (product_matrix_C, witness_matrix_W).
    
    Note: The witness finding part remains O(n^3) in this implementation,
          as it explicitly iterates through 'k' to find a witness.
          This function does not use Strassen for its internal product calculation
          as the witness finding logic would need to be integrated differently.
          It uses the standard Boolean multiplication to find the witnesses.
    """
    if not isinstance(A, Matrix) or not isinstance(B, Matrix):
        raise TypeError("Inputs must be Matrix objects.")

    rows_A, cols_A = A.get_dimensions()
    rows_B, cols_B = B.get_dimensions()

    if cols_A != rows_B:
        raise ValueError(
            f"Number of columns in A ({cols_A}) must equal "
            f"number of rows in B ({rows_B}) for Boolean matrix multiplication."
        )

    product_data_C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    witness_data_W = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            found_witness = False
            for k in range(cols_A): # k iterates through the 'middle' dimension
                if A[i][k] == 1 and B[k][j] == 1:
                    product_data_C[i][j] = 1
                    witness_data_W[i][j] = k + 1 # Store k+1 to differentiate from 0 (no witness)
                    found_witness = True
                    break
            if not found_witness:
                witness_data_W[i][j] = 0

    return Matrix(product_data_C), Matrix(witness_data_W)

# Basic test if bmm_witnesses.py is run directly
if __name__ == "__main__":
    print("--- Testing Boolean Matrix Multiplication ---")
    A_data = [[1, 0, 1],
              [0, 1, 0],
              [1, 1, 0]]
    B_data = [[0, 1, 0],
              [1, 0, 1],
              [0, 1, 1]]
    A = Matrix(A_data)
    B = Matrix(B_data)

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)

    C_boolean_standard = boolean_matrix_multiply(A, B, use_strassen=False)
    print("\nBoolean Product C (Standard):")
    print(C_boolean_standard)

    C_boolean_strassen = boolean_matrix_multiply(A, B, use_strassen=True)
    print("\nBoolean Product C (via Strassen):")
    print(C_boolean_strassen)
    
    print(f"Results are equal (small matrix): {C_boolean_standard == C_boolean_strassen}")

    print("\n--- Testing BMM Witness Algorithm ---")
    # Note: bmm_witness itself uses the standard O(n^3) loop for finding witnesses.
    # The 'use_strassen_for_product' parameter was removed as it's not applicable
    # to the inner witness-finding logic in this current design.
    C_witness, W_matrix = bmm_witness(A, B) 
    print("Boolean Product C (from bmm_witness):")
    print(C_witness)
    print("Witness Matrix W (0 for no witness, k+1 for witness k):")
    print(W_matrix)

    # --- Runtime Experiments: Standard vs. Strassen for Boolean Matrix Multiplication ---
    print("\n--- Runtime Experiments: Standard vs. Strassen for Boolean Matrix Multiplication ---")
    
    # Define thresholds for Strassen's; these often need tuning
    # A common starting point is around n=16 to 32 for the base case.
    STRASSEN_THRESHOLD = 32 

    # Test 1: Small matrix (Strassen might be slower due to overhead)
    print(f"\nTest 1: Matrix size 32x32 (n={32})")
    n = 32
    A_exp = Matrix([[1 if (i+j)%2==0 else 0 for j in range(n)] for i in range(n)])
    B_exp = Matrix([[1 if (i*j)%3==0 else 0 for j in range(n)] for i in range(n)])

    start_time = time.perf_counter()
    res_standard = boolean_matrix_multiply(A_exp, B_exp, use_strassen=False)
    end_time = time.perf_counter()
    print(f"Standard BMM (n={n}): {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    res_strassen = boolean_matrix_multiply(A_exp, B_exp, use_strassen=True, strassen_threshold=STRASSEN_THRESHOLD)
    end_time = time.perf_counter()
    print(f"Strassen BMM (n={n}): {end_time - start_time:.6f} seconds")
    print(f"Results are equal: {res_standard == res_strassen}")

    # Test 2: Medium matrix (power of 2)
    print(f"\nTest 2: Matrix size 128x128 (n={128})")
    n = 128
    A_exp = Matrix([[1 if (i+j)%2==0 else 0 for j in range(n)] for i in range(n)])
    B_exp = Matrix([[1 if (i*j)%3==0 else 0 for j in range(n)] for i in range(n)])

    start_time = time.perf_counter()
    res_standard = boolean_matrix_multiply(A_exp, B_exp, use_strassen=False)
    end_time = time.perf_counter()
    print(f"Standard BMM (n={n}): {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    res_strassen = boolean_matrix_multiply(A_exp, B_exp, use_strassen=True, strassen_threshold=STRASSEN_THRESHOLD)
    end_time = time.perf_counter()
    print(f"Strassen BMM (n={n}): {end_time - start_time:.6f} seconds")
    print(f"Results are equal: {res_standard == res_strassen}")

    # Test 3: Larger matrix (power of 2)
    print(f"\nTest 3: Matrix size 256x256 (n={256})")
    n = 256
    A_exp = Matrix([[1 if (i+j)%2==0 else 0 for j in range(n)] for i in range(n)])
    B_exp = Matrix([[1 if (i*j)%3==0 else 0 for j in range(n)] for i in range(n)])

    start_time = time.perf_counter()
    res_standard = boolean_matrix_multiply(A_exp, B_exp, use_strassen=False)
    end_time = time.perf_counter()
    print(f"Standard BMM (n={n}): {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    res_strassen = boolean_matrix_multiply(A_exp, B_exp, use_strassen=True, strassen_threshold=STRASSEN_THRESHOLD)
    end_time = time.perf_counter()
    print(f"Strassen BMM (n={n}): {end_time - start_time:.6f} seconds")
    print(f"Results are equal: {res_standard == res_strassen}")

    # Test 4: What happens with non-power-of-2 dimensions?
    print(f"\nTest 4: Matrix size 100x100 (non-power-of-2, n={100})")
    n = 100
    A_exp = Matrix([[1 if (i+j)%2==0 else 0 for j in range(n)] for i in range(n)])
    B_exp = Matrix([[1 if (i*j)%3==0 else 0 for j in range(n)] for i in range(n)])

    start_time = time.perf_counter()
    res_standard = boolean_matrix_multiply(A_exp, B_exp, use_strassen=False)
    end_time = time.perf_counter()
    print(f"Standard BMM (n={n}): {end_time - start_time:.6f} seconds")

    start_time = time.perf_counter()
    res_strassen = boolean_matrix_multiply(A_exp, B_exp, use_strassen=True, strassen_threshold=STRASSEN_THRESHOLD)
    end_time = time.perf_counter()
    print(f"Strassen BMM (n={n}): {end_time - start_time:.6f} seconds")
    print(f"Results are equal: {res_standard == res_strassen}")