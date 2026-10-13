# ZView, a Zephyr RTOS runtime visualizer

Real-time system observability for Zephyr RTOS, delivered over SWD.

**Stop guessing your stack margins and heap health.** ZView provides a zero-footprint, high-fidelity visualization of your Zephyr application's runtime, delivered over SWD with zero instrumentation.

*No UART, no RTT, and no manual code changes. Just Kconfig and your probe.*

> **This page covers the standalone pip installation.** Everything else, including using ZView from a west workspace, is in the [main documentation](https://github.com/wkhadgar/zview/blob/main/README.md).


## Installation

```
pip install zview
```

Or from the root of the repository:

```
pip install .
```


## Prerequisites

Any Zephyr build works, and a few Kconfig options add to what ZView can show. The `zview` snippet that sets them ships with the west module, not with the pip package, so set them in your `prj.conf` as listed in the [prerequisites](https://github.com/wkhadgar/zview/blob/main/README.md#prerequisites).


## How to Use

```
# ELF, runner and target read from the build directory
zview -d build

# Or passed explicitly, e.g. nRF5340 DK via JLink
zview -e build/zephyr/zephyr.elf -r jlink -t nRF5340_xxAA
```

`zview --help` and `zview <command> --help` list every command and argument. The [main documentation](https://github.com/wkhadgar/zview/blob/main/README.md#commands) covers what each one does, how to pick `-t` for your runner, navigation, and offline recording and replay.
