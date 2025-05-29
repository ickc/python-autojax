This is a set of experiments testing JAX's capability to use multiple CPU cores.

Experiment can be run with

```bash
pixi run profile.sh
```

Summary of lesson learnt:

- `XLA_FLAGS='--xla_force_host_platform_device_count=...'` is effectively the same as `JAX_NUM_CPU_DEVICES=...`.
- `XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'` seems to have no effect anymore. Use `NPROC=1` to disable multithreading used in eigen instead.
