# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.2.0] 2026-11-02

### Performance

- Numpy autodiff engine with the same hand-written gradients (#3)

### Documentation

- Standard README structure and language framing (#12)
- Horizontal layered architecture diagram
- Compact horizontal architecture diagram
- Remove authorship statements from docs and scripts
- Recorded demos of more commands, with their vhs tapes (#1)
- Recorded demos of the HTTP service and more commands (#2)

### Build and CI

- Run CodeQL on main and on a weekly schedule only

### Maintenance

- Add release automation, AGENTS.md, llms.txt and contributor docs
- Classify squash merges in the changelog by the commits they contain

## [0.1.0] 2026-11-02

### Features

- Implement autodiff engine and transformer language model (#1)
- Real-text training, checkpointing, and PyTorch numerical parity (#9)

### Bug fixes

- Feasible data splits for small tasks and empty-sequence guard (#7)

### Documentation

- Remove em dashes from documentation (#10)

### Maintenance

- Initialize repository with license and overview
- Bump GitHub Actions to current major versions (#5)
- Remove Dependabot configuration
