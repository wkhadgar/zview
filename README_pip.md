# ZView, a Zephyr RTOS runtime visualizer

Zephyr RTOS system-wide runtime visualizer via SWD probe!

Take a broader look on your Zephyr application with a non-heavy, small footprint, Kconfig-only thread stats analyser.

> **This page covers the standalone pip installation.** If you are using ZView inside a west workspace (the most common setup), see the [main documentation](https://github.com/wkhadgar/zview/blob/main/README.md) instead.


## Prerequisites

To properly analyze your Zephyr app, your ELF binary must be compiled with specific Kconfig options enabled:

```
## prj.conf
CONFIG_INIT_STACKS=y            # Required for stack watermarks
CONFIG_THREAD_MONITOR=y         # Required for thread discovery
CONFIG_THREAD_STACK_INFO=y      # Required for thread metadata

# Optional Features
CONFIG_THREAD_NAME=y            # Enables thread name display
CONFIG_THREAD_RUNTIME_STATS=y   # Enables CPU usage tracking
CONFIG_SYS_HEAP_RUNTIME_STATS=y # Enables heap runtime stats
```


## Installation

Install ZView in your Python virtual environment:

```
pip install zview
```

Or from the root of the repository:

```
pip install .
```


## How to Use

```
# Example: nRF5340 DK via JLink
zview -e build/zephyr/zephyr.elf -r jlink -t nRF5340_xxAA
```


## Arguments

| Argument | Description |
| --- | --- |
| `-e, --elf-file` | Path to the firmware `.elf` file (e.g. `build/zephyr/zephyr.elf`). |
| `-r, --runner` | Debug runner to use: `jlink`, `pyocd`, or `gdb`. |
| `-t, --runner-target` | MCU descriptor for the chosen runner (see below). |
| `--period` | Update period in seconds, can be a float. |

<details>
<summary><strong>Finding the right value for <code>-t</code></strong></summary>
<br>

The `-t` argument is the MCU descriptor name **as expected by your chosen runner**. How to find it depends on the runner:

**JLink (`-r jlink`)**
Use the device name from the [J-Link Supported Devices list](https://www.segger.com/downloads/supported-devices.php).
```
# Example: Nordic nRF5340 DK
zview -e build/zephyr/zephyr.elf -r jlink -t nRF5340_xxAA
```

**pyOCD (`-r pyocd`)**
Run `pyocd list --targets` to see the available target names for your installed packs.
```
pyocd list --targets          # find your target name

# Example: STM32F401 Nucleo
zview -e build/zephyr/zephyr.elf -r pyocd -t stm32f401xe
```

**GDB server (`-r gdb`)**
Pass the `host:port` of your GDB server instead of a device name.
```
zview -e build/zephyr/zephyr.elf -r gdb -t localhost:1234
```

</details>

---

For navigation, advanced usage, and QEMU/GDB targets, refer to the [main documentation](https://github.com/wkhadgar/zview/blob/main/README.md).
