from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

bench_enet = Benchmark('./')

run_benchmark(bench_enet, solver_names=["fista"], dataset_names=["2d-mri"], max_runs=20,
                  n_repetitions=3, timeout=1000)

#  benchopt run -s sklearn -d leukemia --max-runs 100 --n-repetitions 5 