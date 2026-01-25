import time
import numpy as np
import pandas as pd


def benchmark_array_growth(n_iterations=1000, dim=1000, batch_size=1):
    """
    Benchmarks different ways of growing a dataset from 0 to n_iterations * batch_size.
    """
    print(
        f"Benchmarking growth to N={n_iterations * batch_size}, D={dim}, batch_size={batch_size}\n"
    )

    results = []

    # --- Method 1: List append + np.asarray() at each step (The "Current" way) ---
    # This is O(N^2) total because each asarray() is O(N)
    t0 = time.perf_counter()
    data_list = []
    times = []
    for i in range(n_iterations):
        new_data = np.random.rand(batch_size, dim)
        data_list.append(new_data.tolist())
        _ = np.asarray(
            data_list, dtype=float
        )  # Simulate the conversion happening inside tell/ask
        times.append(time.perf_counter() - t0)
    results.append(("List + asarray()", times[-1], "O(N^2) total"))

    # --- Method 2: np.concatenate() at each step (The "Skeptical" way) ---
    # This is also O(N^2) total because concatenate() copies the whole array
    t0 = time.perf_counter()
    data_arr = None
    for i in range(n_iterations):
        new_data = np.random.rand(batch_size, dim)
        if data_arr is None:
            data_arr = new_data
        else:
            data_arr = np.concatenate([data_arr, new_data], axis=0)
    results.append(("np.concatenate()", time.perf_counter() - t0, "O(N^2) total"))

    # --- Method 3: List append + np.vstack() ONLY at the end ---
    # This is O(N) total
    t0 = time.perf_counter()
    data_list = []
    for i in range(n_iterations):
        new_data = np.random.rand(batch_size, dim)
        data_list.append(new_data)
    _ = np.vstack(data_list)
    results.append(
        ("List append (final vstack)", time.perf_counter() - t0, "O(N) total")
    )

    # --- Method 4: Pre-allocated buffer (The "High Performance" way) ---
    # This is O(N) total and avoids most allocations
    t0 = time.perf_counter()
    capacity = 100  # Initial capacity
    buffer = np.empty((capacity, dim))
    size = 0
    for i in range(n_iterations):
        new_data = np.random.rand(batch_size, dim)
        if size + batch_size > capacity:
            capacity *= 2  # Exponential growth
            new_buffer = np.empty((capacity, dim))
            new_buffer[:size] = buffer[:size]
            buffer = new_buffer
        buffer[size : size + batch_size] = new_data
        size += batch_size
    _ = buffer[:size]
    results.append(("Pre-allocated Buffer", time.perf_counter() - t0, "O(N) total"))

    # Display results
    df = pd.DataFrame(results, columns=["Method", "Total Time (s)", "Complexity"])
    print(df.to_string(index=False))

    # Calculate speedup relative to Method 1
    baseline = results[0][1]
    print(f"\nSpeedup (Buffer vs List+asarray): {baseline / results[3][1]:.1f}x")


if __name__ == "__main__":
    # Test with N=2000, D=1000.
    # Total floats = 2 million.
    benchmark_array_growth(n_iterations=2000, dim=1000, batch_size=1)
