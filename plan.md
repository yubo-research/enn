
Find ways to speed up `./ops/stress.py enn bpann_disk 1000000 --num-dim=1000 --work-dir=_enn/`.


A basline run produced:
```
num_dim=1000 num_obs=1000000 work_dir=_enn/
      1 0.004 8.7e-05
      3 0.006 7.71e-05
     10 0.018 0.000127
     30 0.023 0.000248
    100 0.035 0.00142
    300 0.066 0.00301
   1000 0.199 0.0101
   3000 0.367 0.0257
  10000 0.444 0.1
  30000 0.387 0.318
 100000 0.398 1.39
heartbeat n=256001
 300000 0.606 9.63
heartbeat n=356737
heartbeat n=422849
heartbeat n=477697
heartbeat n=522625
heartbeat n=561985
heartbeat n=598081
heartbeat n=631873
heartbeat n=662721
heartbeat n=692609
heartbeat n=721473
heartbeat n=747713
heartbeat n=774145
heartbeat n=799041
heartbeat n=823297
heartbeat n=847425
heartbeat n=871233
heartbeat n=894529
heartbeat n=917505
heartbeat n=940033
heartbeat n=961665
heartbeat n=983105
1000000 0.868 214
```

The goal is to reduce the time between updates (the third column
 in the non-heartbeat rows of the sample output above). Although,
 we only care about the time spent in the ennbo library. We don't care
 about the time spent generating the test rng data, for example. You
 might need to add a separate timer to get at that.

# TODO

- Add a --batch option to `./ops/stress.py enn`. When activateded,
 permit stress.py to add multiple observations to ENN at once. Add
 support for this where required.
- Run `rm -rf _enn; ./ops/stress.py enn --batch bpann_disk 1000000 --num-dim=1000 --work-dir=_enn/` to
   evaluate the change.
- Iterate until the third-column times are all ~3x smaller than the baseline run's.
