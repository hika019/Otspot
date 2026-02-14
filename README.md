# solver

A high-performance mathematical optimization solver written in Rust.

## Overview

`solver` is a next-generation optimization solver designed to tackle linear programming (LP), mixed-integer programming (MIP), and eventually nonlinear programming (NLP) problems. Built with Rust, it prioritizes performance, memory safety, and parallelization-first architecture.

The project follows a staged expansion strategy: starting with LP solvers, extending to MIP with advanced parallel capabilities, and ultimately targeting NLP with GPU acceleration.

## Vision

- **Performance target**: Achieve 80%+ of HiGHS performance in Phase 1 (small to medium-scale LP problems)
- **Parallelization-first design**: Leverage Rust's memory safety to enable efficient parallel tree search (a key differentiator in Phase 2)
- **Ecosystem integration**: Aim for PyPI publication and SciPy integration to reach the broad scientific computing community

## Current Status

**Phase 1 M1**: Primal Simplex MVP (in development)

The project is currently in the early development phase, focusing on building the foundational LP solver with Primal Simplex method.

## Features (Planned)

### Phase 1 (0-12 months): LP Simplex MVP
- Primal Simplex (M1, 0-3 months)
- Dual Simplex (M2, 3-6 months)
- Basic preprocessing
- Python bindings via PyO3 (M3, 6-9 months)
- PyPI publication and SciPy PR submission (M4, 9-12 months)

### Phase 2 (12-24 months): MIP with Parallelization
- Branch-and-cut algorithm
- Parallel tree search (leveraging Rust's safety guarantees)
- Cutting planes (Gomory cuts, MIR, etc.)
- Target: 90%+ of HiGHS performance on MIPLIB 2017

### Phase 3 (24-36 months): NLP/GPU Extension
Direction to be determined based on Phase 1-2 user feedback:
- Option A: GPU-accelerated LP/MIP (PDHG-based)
- Option B: NLP solvers (SQP, interior-point methods)
- Option C: Performance optimization to commercial-grade levels

## Building

```bash
cargo build
cargo test
```

## License

This project is dual-licensed under:

- **Apache License 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
- **MIT License** ([LICENSE-MIT](LICENSE-MIT))

You may choose either license for your use.

## Roadmap

For detailed strategic planning, technical decisions, and milestone definitions, see the project documentation in the repository.

## Contributing

Contributions are welcome! This project is in its early stages, and we're building the foundation together.
