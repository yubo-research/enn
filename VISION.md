


- Python exposes the public API to Python users, but all the work is in Rust.
- The running time of ENN adds and queries grows as k*D*lnN (or better). A good teat case is k=10, D=1000, N grows from 1 to 1e5.  (k is number of neighbors, D is number of dimensions, N is number of observations).
- Memory is O(kD) (or better), constant in N.
