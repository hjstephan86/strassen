# experiments.py

import time
import random
import matplotlib.pyplot as plt
import numpy as np # For easier random matrix generation
import psutil # For system monitoring

# Import our custom Matrix class and Strassen's algorithm
from matrix import Matrix
from strassen import strassen_multiply

# --- Helper function to print system usage ---
def print_system_usage(stage="", file=None):
    """
    Prints current system-wide CPU and RAM usage to the console and optionally to a file.
    """
    cpu_percent = psutil.cpu_percent(interval=0.1) # CPU usage over 0.1 seconds
    mem_info = psutil.virtual_memory()

    output_lines = []
    output_lines.append(f"\n--- System Usage {stage} ---")
    output_lines.append(f"CPU Usage: {cpu_percent:.1f}%")
    output_lines.append(f"Total RAM: {mem_info.total / (1024**3):.2f} GB")
    output_lines.append(f"Available RAM: {mem_info.available / (1024**3):.2f} GB")
    output_lines.append(f"Used RAM: {mem_info.used / (1024**3):.2f} GB")
    output_lines.append(f"--- End System Usage ---\n")

    for line in output_lines:
        print(line) # Print to console
        if file:
            file.write(line + "\n") # Write to file if a file object is provided

def generate_random_matrix(n, val_range=(0, 100)):
    """
    Generates a random n x n Matrix object with integer values.
    """
    data = [[random.randint(val_range[0], val_range[1]) for _ in range(n)] for _ in range(n)]
    return Matrix(data)

def run_experiment(max_n, step=4, num_trials=3, threshold_for_strassen=16, output_filename="results.txt"):
    """
    Conducts experiments to compare standard vs. Strassen multiplication.
    Writes results to both the console and a specified output file.

    Args:
        max_n (int): The maximum matrix dimension to test.
        step (int): Increment for matrix dimension (e.g., 4, 8, 16).
        num_trials (int): Number of times to run each multiplication for averaging.
        threshold_for_strassen (int): The threshold passed to strassen_multiply.
                                      This influences Strassen's performance.
        output_filename (str): The name of the file to write results to.

    Returns:
        tuple: (list of n_values, list of avg_standard_times, list of avg_strassen_times)
    """
    n_values = []
    standard_times = []
    strassen_times = []

    # Open the output file for writing (will create or overwrite it)
    with open(output_filename, 'w') as f:
        print_system_usage(stage="BEFORE EXPERIMENT START", file=f) # ADDED: Log usage at start to console and file

        # Prepare header information for both console and file output
        header_lines = [
            f"Starting experiments up to n={max_n}, step={step}, trials={num_trials}",
            f"Strassen's internal threshold set to: {threshold_for_strassen}",
            f"-" * 50,
            f"{'n':<8} | {'Standard Avg Time (s)':<22} | {'Strassen Avg Time (s)':<22}",
            f"-" * 50
        ]
        for line in header_lines:
            print(line) # Print to console
            f.write(line + "\n") # Write to file

        # Start n from a small value, or the step size if it's larger
        current_n = max(1, step)
        
        while current_n <= max_n:
            n_values.append(current_n)
            
            # --- Measure Standard Multiplication Time ---
            total_standard_time = 0
            for _ in range(num_trials):
                A = generate_random_matrix(current_n)
                B = generate_random_matrix(current_n)
                start_time = time.perf_counter()
                _ = A * B # Use standard multiplication (__mul__)
                end_time = time.perf_counter()
                total_standard_time += (end_time - start_time)
            avg_standard_time = total_standard_time / num_trials
            standard_times.append(avg_standard_time)

            # --- Measure Strassen Multiplication Time ---
            total_strassen_time = 0
            for _ in range(num_trials):
                A = generate_random_matrix(current_n)
                B = generate_random_matrix(current_n)
                start_time = time.perf_counter()
                _ = strassen_multiply(A, B, threshold=threshold_for_strassen)
                end_time = time.perf_counter()
                total_strassen_time += (end_time - start_time)
            avg_strassen_time = total_strassen_time / num_trials
            strassen_times.append(avg_strassen_time)

            result_line = f"{current_n:<8} | {avg_standard_time:<22.6f} | {avg_strassen_time:<22.6f}"
            print(result_line) # Print to console
            f.write(result_line + "\n") # Write to file

            # Determine next n:
            # If current_n is 1, next is 2
            # If current_n is a power of 2, next is current_n + step OR current_n * 2 to get to next power of 2 quickly
            # We want to primarily test powers of 2 for Strassen's direct benefit, but also intermediate points
            
            # Simple step increment, will cover some non-powers of 2, which is good for padding test
            current_n += step 
            
            # Or uncomment below for a more aggressive power-of-2 stepping, might miss some n_0
            # if current_n < 4: current_n = 4 # Start powers of 2 from at least 4
            # else: current_n *= 2 # Jump to next power of 2

        footer_line = f"-" * 50
        print(footer_line) # Print to console
        f.write(footer_line + "\n") # Write to file

        print_system_usage(stage="AFTER EXPERIMENT FINISH", file=f) # ADDED: Log usage at end to console and file
    
    print(f"\nExperiment results also saved to '{output_filename}'") # Inform user about file saving
    return n_values, standard_times, strassen_times

def plot_results(n_values, standard_times, strassen_times):
    """
    Plots the comparison results and tries to estimate n0.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, standard_times, label='Standard Multiplication', marker='o')
    plt.plot(n_values, strassen_times, label='Strassen\'s Algorithm', marker='x')

    plt.xlabel('Matrix Dimension (n)')
    plt.ylabel('Average Time (seconds)')
    plt.title('Performance Comparison: Standard vs. Strassen Matrix Multiplication')
    plt.legend()
    plt.grid(True)
    plt.xscale('log') # Log scale for n can be useful for larger ranges
    plt.yscale('log') # Log scale for time often shows polynomial growth better

    # Estimate n0
    n0 = None
    for i in range(len(n_values)):
        if strassen_times[i] < standard_times[i]:
            n0 = n_values[i]
            # Try to get a more precise n0 if it's the first time Strassen is faster
            if i > 0 and strassen_times[i-1] >= standard_times[i-1]:
                # Interpolate if possible, otherwise just take current n_values[i]
                n0_lower = n_values[i-1]
                n0_upper = n_values[i]
                print(f"\nApproximate crossover point (n0) found between n={n0_lower} and n={n0_upper}")
                # A more precise interpolation is complex and might not be reliable
                # due to discrete steps and measurement noise.
            else:
                print(f"\nApproximate crossover point (n0) starts around n={n0}")
            break
    
    if n0 is not None:
        plt.axvline(x=n0, color='r', linestyle='--', label=f'Approx. Crossover n0={n0}')
        plt.text(n0, plt.ylim()[1]*0.8, f'n0={n0}', color='r', ha='left', va='center', rotation=90)
        plt.legend()

    plt.tight_layout()

    # plt.savefig('strassen_performance_comparison.png')
    plt.savefig('strassen_performance_comparison.svg')
    print("\nPlot saved as 'strassen_performance_comparison.svg'")
    
    plt.show()

if __name__ == "__main__":
    # --- Configuration for the experiment ---
    # Max matrix dimension to test. Start smaller if it takes too long.
    # On typical machines, n=512 or 1024 might take a few minutes.
    # My local testing shows ~n=128 is a good max_n for a quick run.
    MAX_MATRIX_DIMENSION = 256
    
    # Step size for matrix dimensions. Smaller step gives more detail, but more runs.
    # Using a larger step (e.g., 4 or 8) is good for initial exploration.
    # If using powers of 2 for steps (e.g., current_n *= 2 in loop), set step=1.
    DIMENSION_STEP = 4 
    
    # Number of times to repeat each multiplication for averaging time
    NUMBER_OF_TRIALS = 5 
    
    # This is the internal threshold for Strassen's algorithm itself.
    # Finding the optimal internal threshold is another experiment!
    # Common values are 16, 32, or 64.
    STRASSEN_INTERNAL_THRESHOLD = 32 # This value significantly impacts Strassen's performance

    # --- Run the experiment ---
    n_values, standard_times, strassen_times = run_experiment(
        max_n=MAX_MATRIX_DIMENSION,
        step=DIMENSION_STEP,
        num_trials=NUMBER_OF_TRIALS,
        threshold_for_strassen=STRASSEN_INTERNAL_THRESHOLD,
        output_filename="results.txt" # Specify the output filename
    )

    # --- Plot the results ---
    plot_results(n_values, standard_times, strassen_times)

    print("\nExperiment finished. Check the plot for n0.")
