# Contributing to Pikafish

Welcome to the Pikafish project! We are excited that you are interested in
contributing. This document outlines the guidelines and steps to follow when
making contributions to Pikafish.

## Table of Contents

- [Building Pikafish](#building-pikafish)
- [Making Contributions](#making-contributions)
  - [Reporting Issues](#reporting-issues)
  - [Submitting Pull Requests](#submitting-pull-requests)
- [Code Style](#code-style)
- [Community and Communication](#community-and-communication)
- [Dependency Management](#dependency-management)
- [License](#license)

## Building Pikafish

In case you do not have a C++ compiler installed, you can follow the
instructions from our wiki.

- [Linux][linux-compiling-link]
- [Windows][windows-compiling-link]
- [macOS][macos-compiling-link]

## Making Contributions

### Reporting Issues

If you find a bug, please open an issue on the
[issue tracker][issue-tracker-link]. Be sure to include relevant information
like your operating system, build environment, and a detailed description of the
problem.

_Please note that Pikafish's development is not focused on adding new features.
Thus any issue regarding missing features will potentially be closed without
further discussion._

### Submitting Pull Requests

- Functional changes need to be tested on fishtest. See
  [Creating my First Test][creating-my-first-test] for more details.
  The accompanying pull request should include a link to the test results and
  the new bench.

- Non-functional changes (e.g. refactoring, code style, documentation) do not
  need to be tested on fishtest, unless they might impact performance.

- Provide a clear and concise description of the changes in the pull request
  description.

_First time contributors should add their name to [AUTHORS](./AUTHORS)._

_Pikafish's development is not focused on adding new features. Thus any pull
request introducing new features will potentially be closed without further
discussion._

## Code Style

Changes to Pikafish C++ code should respect our coding style defined by
[.clang-format](.clang-format). You can format your changes by running
`make format`. This requires clang-format version 20 to be installed on your system.

## Community and Communication

- Join the [Pikafish discord][discord-link] to discuss ideas, issues, and
  development.
- Participate in the [Pikafish GitHub discussions][discussions-link] for
  broader conversations.

## License

By contributing to Pikafish, you agree that your contributions will be licensed
under the GNU General Public License v3.0. See [Copying.txt][copying-link] for
more details.

## Dependency Management

Pikafish vendors its native dependencies rather than relying on a system package
manager. This keeps the build self-contained and reproducible. The only
automated dependency tracking is for GitHub Actions (via
[Dependabot](.github/dependabot.yml)); the C/C++ libraries below are synced
manually.

### zstd (vendored under `src/external/`)

zstd is embedded to decompress NNUE net files at runtime. Only the **decompression**
subset of the library is vendored (no compression code).

**Update procedure:**

1. Download a new release from <https://github.com/facebook/zstd> (tagged
   release tarball).
2. Copy **only** the decompression-related sources into `src/external/`:
   - `lib/common/*`           → `src/external/common/`
   - `lib/decompress/*`       → `src/external/decompress/`
   - `lib/zstd.h`, `lib/zstd_errors.h` → `src/external/`
   - Keep the amd64 assembly (`huf_decompress_amd64.S`) in sync with the
     matching C sources.
3. Rebuild (`make -j build` from `src/`) and run `./pikafish bench`.
4. **Verify the bench node count is unchanged** — this confirms the decompressor
   still produces byte-identical nets. If the node count differs, do not ship
   the update; investigate the regression first.

### pybind11 (toolchain only)

pybind11 is used solely by `tools/nnue_pybind.cpp` to expose the NNUE evaluator
to Python for training/verification scripts. It is **not** linked into the
`pikafish` engine binary.

**Update procedure:**

1. Download the single-header distribution from
   <https://github.com/pybind/pybind11> (or the full source tree).
2. Replace the header(s) on the include path used by
   `tools/build_nnue_pybind.sh`.
3. Rebuild the binding with `bash tools/build_nnue_pybind.sh` and run
   `python tools/test_nnue_pybind.py` to confirm it still compiles and the
   numerical tests pass.

### GoogleTest (test framework)

The unit-test framework is fetched on demand via CMake `FetchContent` (see the
test configuration). There is **no vendored copy** and no manual update step —
the version is pinned in the test CMake configuration and refreshed by
re-running CMake configuration.

### GitHub Actions

CI action versions (`actions/checkout`, `actions/upload-artifact`,
`actions/download-artifact`, `msys2/setup-msys2`,
`jidicula/clang-format-action`, `github/codeql-action`, …) are tracked
automatically by Dependabot (`.github/dependabot.yml`). Review the weekly
pull requests and bump after confirming CI is green.

Thank you for contributing to Pikafish and helping us make it even better!

[copying-link]: https://github.com/official-pikafish/Pikafish/blob/master/Copying.txt
[discord-link]: https://discord.com/invite/uSb3RXb7cY
[discussions-link]: https://github.com/official-pikafish/Pikafish/discussions/new
[creating-my-first-test]: https://github.com/glinscott/fishtest/wiki/Creating-my-first-test#create-your-test
[issue-tracker-link]: https://github.com/official-pikafish/Pikafish/issues
[linux-compiling-link]: https://github.com/official-pikafish/Pikafish/wiki/Compiling-from-source#linux
[windows-compiling-link]: https://github.com/official-pikafish/Pikafish/wiki/Compiling-from-source#windows
[macos-compiling-link]: https://github.com/official-pikafish/Pikafish/wiki/Compiling-from-source#macos
