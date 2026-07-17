
Treat this as the eval function, f(x): `./ops/stress.py enn bpann_disk 1000000 --num-dim=1000 --work-dir=_enn/`.
- Each output row (excluding heartbeat rows) has the form "N t(N) s(N)", where t(N) & s(N) are times. Consider the
   t(N) and s(N) the vector output of f(x). We're interested in decrease all of them to some extent.

In ~/.ennbo/config.toml are the default parameters, the vector x. Sweep each x_i, one at a time leave all other elements
 of x at their default values. Record all f(x) outputs for all sweeps in bpann_sweeps.md.

Append a short report of your findings to the bottom of bpann_sweeps.md.
