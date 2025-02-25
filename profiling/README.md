This is a set of experiments testing various ways to perform reduced sum over vector(s) of length $K$ without expanding to higher dimensional N-D arrays that consumes memory proportional to $K$.

I.e. $F(\ldots, v, \ldots) = \sum_k f(\ldots, v_k, \ldots)$

Different equivalent ways to do the same thing in JAX are,

1. explicit vectorization: expand the $K$-dimension as outer products and perform matmul to reduce the sum;
2. use `vmap` to vectorize a function that acts on the k-th elements only, and apply `.sum` to reduce it;
3. use `scan` to perform a scan (a.k.a. prefix sum) operation while not keeping the intermediate values;
4. use `fori_loop` to perform a loop over effectively `k in range(K)` explicitly;
5. use `pmap`.

Experiment can be run with

```bash
pixi run profile.sh
```

Summary of lesson learnt:

- explicit vectorization and `vmap` are almost identical in performance and memory use, with (2) having slight edge on both metrics.
- `pmap` is actually map over devices, not over cores/threads for example. This is irrelevant for the current context.
- `scan` and `fori_loop` has virtually identical performance and memory use, which is superior than other methods both in speed and memory need. Memory use does not depends on $K$, which is the most important advantage.
- `scan` has a slight edge on semantics, as it can be written in a way very similar to `vmap`. See code (`profile_jax_vmap.py` and `profile_jax_scan.py`, etc. for examples).
