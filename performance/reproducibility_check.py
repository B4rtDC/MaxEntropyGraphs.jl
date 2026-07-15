"""Reproducibility check for NEMtropy's ensemble sampler.

Verifies that a fixed seed reproduces the same sample, and that this holds independent of the
number of cores (cpu_n). The MaxEntropyGraphs.jl side is covered by the package's own test
suite ("sampling reproducibility" testset in test/solver.jl), which shows that
`rand(model, n; rng=Xoshiro(s))` is reproducible and thread-count-independent.

Run under the uv venv:  ./.venv/bin/python reproducibility_check.py
"""
import hashlib
import os
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore")
import networkx as nx
from NEMtropy import UndirectedGraph


def _digest(directory):
    h = hashlib.sha256()
    for fn in sorted(os.listdir(directory)):
        with open(os.path.join(directory, fn), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def check(cpu_n, seed=42, n=5):
    g = nx.karate_club_graph()
    edgelist = [(u, v) for u, v in g.edges()]
    base = tempfile.mkdtemp()
    digests = []
    for run in ("a", "b"):
        d = os.path.join(base, f"{run}{cpu_n}")
        M = UndirectedGraph(edgelist=edgelist)
        M.solve_tool(model="cm_exp", method="fixed-point", initial_guess="degrees")
        M.ensemble_sampler(n, cpu_n=cpu_n, output_dir=d + "/", seed=seed)
        digests.append(_digest(d))
    shutil.rmtree(base)
    return digests[0], digests[1], digests[0] == digests[1]


if __name__ == "__main__":
    ok = True
    for cpu_n in (1, 4):
        a, b, same = check(cpu_n)
        print(f"NEMtropy cpu_n={cpu_n}: seed reproducible = {same} (digest {a})")
        ok = ok and same
    print("ALL REPRODUCIBLE" if ok else "NOT REPRODUCIBLE")
