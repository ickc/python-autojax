# Notes on running benchmark

```sh
% pixi run pytest-benchmark list
# these are run on cosma5
.benchmarks/Linux-CPython-3.13-64bit/0001_N=30_B=3_K=1024_P=32_S=256_NUM_THREADS=1.json
.benchmarks/Linux-CPython-3.13-64bit/0002_N=32_B=3_K=8192_P=32_S=256_NUM_THREADS=256.json
.benchmarks/Linux-CPython-3.13-64bit/0003_N=32_B=3_K=8192_P=32_S=256_NUM_THREADS=1.json
.benchmarks/Linux-CPython-3.13-64bit/0004_N=64_B=3_K=32768_P=32_S=256_NUM_THREADS=256.json
.benchmarks/Linux-CPython-3.13-64bit/0005_N=64_B=3_K=32768_P=32_S=256_NUM_THREADS=1.json
# these are symlinks to make comparison easier to read
.benchmarks/Linux-CPython-3.13-64bit/1_N=64_B=3_K=32768_P=32_S=256_NUM_THREADS=1.json
.benchmarks/Linux-CPython-3.13-64bit/256_N=64_B=3_K=32768_P=32_S=256_NUM_THREADS=256.json
# e.g.
pixi run pytest-benchmark compare 1_N=64_B=3_K=32768_P=32_S=256_NUM_THREADS=1 256_N=64_B=3_K=32768_P=32_S=256_NUM_THREADS=256 --columns=mean,stddev,ops,rounds,iterations --sort=mean
```

