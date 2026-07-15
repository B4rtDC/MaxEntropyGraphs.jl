# Contributing to MaxEntropyGraphs.jl

Thanks for your interest in contributing! Contributions of all kinds are welcome:
bug reports, feature requests, documentation improvements, and code.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting issues

Please use the [GitHub issue tracker](https://github.com/B4rtDC/MaxEntropyGraphs.jl/issues)
to report bugs or request features. When reporting a bug, a minimal reproducible
example is enormously helpful. Useful information to include:

- What you did, what you expected to happen, and what actually happened.
- The output of `versioninfo()` and `]status MaxEntropyGraphs` (Julia and package
  versions).
- A minimal code snippet that triggers the problem, and the full error message /
  stack trace.

## Asking questions / getting help

For usage questions, please open a [GitHub Discussion](https://github.com/B4rtDC/MaxEntropyGraphs.jl/discussions)
(or an issue if Discussions are not enabled). The
[documentation](https://B4rtDC.github.io/MaxEntropyGraphs.jl/dev/) is the best
starting point and includes per-model guides and a full API reference.

## Contributing code

1. Fork the repository and create a branch off `main` for your change.
2. Make your change, following the style of the surrounding code.
3. Add or update tests in `test/` so the new behaviour is covered.
4. Make sure the test suite passes locally (see below).
5. Update the documentation (`docs/`) and `CHANGELOG.md` where relevant.
6. Open a pull request describing the change and the motivation behind it.

New contributions should keep the package's public API documented with
docstrings and, where it adds a user-facing capability, include an example in the
documentation.

## Running the tests

From the repository root:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

or, without activating the project first:

```julia
using Pkg
Pkg.test("MaxEntropyGraphs")
```

The suite also runs automatically on every pull request via GitHub Actions across
Julia 1.10 (LTS), 1.11, and 1.12 on Linux, macOS, and Windows, and includes
[Aqua.jl](https://github.com/JuliaTesting/Aqua.jl) quality checks.

## Development tips

- The (expensive) precompilation workload can be disabled while developing by
  setting `precompile_workload = false` in a `LocalPreferences.toml` next to the
  active project. This greatly speeds up iterative reloading.
- Performance-sensitive changes can be checked against the reproducible
  benchmark harness in the [`performance/`](performance/) directory.

Thank you for helping improve MaxEntropyGraphs.jl!
