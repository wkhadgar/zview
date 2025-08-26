# ZView, a Zephyr RTOS runtime visualizer

Zephyr RTOS system-wide runtime visualizer via SWD probe!

Take a broader look on your Zephyr application with a non-heavy, small footprint, Kconfig-only thread stats analyser.

## How to use

To be able to properly analyze your Zephyr app, your ELF binary must be compiled and flashed with just some Kconfig
extra options:

```conf
## prj.conf

CONFIG_INIT_STACKS=y           # For stack watermarks
CONFIG_THREAD_NAME=y           # For thread names
CONFIG_THREAD_MONITOR=y        # For thread search
CONFIG_THREAD_STACK_INFO=y     # For thread information
CONFIG_THREAD_RUNTIME_STATS=y  # For CPU usage tracking
```

You will also need a debug probe capable of SWD debugging, currently only Segger JLink and PyOCD compatible probes are
supported, but more are to come!

Then just run ZViewTUI (a terminal front-end for ZView) on your Zephyr env:

```shell
(.venv) $ python zview_tui.py -e path/to/your/elf --mcu YOUR_MCU_TAG_NAME --runner YOUR_RUNNER
```

> [!NOTE]
> The MCU tag name is the name attributed to your MCU for runner connection, and is optional on some runners.

****

## How it works

ZView attempts to be a neat tool for those who need to debug applications without the need of a shell/UART for stack
statistics. The intent is to achieve minimal footprint and behaviour branching for the binary, avoiding statistics
stdout and processing threads.

The keypoint is to take advantage of the debug probes ability to read runtime memory without CPU halting from the APB
bus, since ELF file contains all information needed about the kernel objects, we just scan through the stack list and
runtime analyse watermarks and CPU usages, so your source code can be kept free from debug implementations.


> [!NOTE]
> When tracking CPU usage, the `idle` thread expresses its usage in true CPU activity, *all* other threads have thier
> CPU usage shown in percentage relative to the *non-idle* time. This means that if a thread `foo` has 50% CPU usage,
> and `idle` has 40%, `foo` is consuming 50% of the remaining 60% of *non-idle* CPU usage, effectively consuming 30% of
> the total CPU execution. This is useful since then it is fast to understand how your threads are behaving in relation
> to other active threads, and how much of the CPU is effectively being used.

****

## Navigation

ZViewTUI acts as a TUI (who would guess?), so you navigate with the arrow keys UP and DOWN from the default view:
![TUI Navigation](./docs/assets/default_view_1.png)
![TUI Navigation example 2](./docs/assets/default_view_2.png)

To track CPU usage for a thread (and return to all threads) just hit ENTER over it:
![Thread tracking](./docs/assets/thread_track_1.png)
![Thread tracking](./docs/assets/thread_track_2.png)


****

### In the future...

Listening to community feedback, ZView is ongoing some new features development, such as, and not limited to:

- Heap visualization
- Extra thread information (like number of context switches and so)
- Full Zephyr integration through West (maybe an in-tree west command?)
- Live global variables tracking
- IDK, feel free to open an [issue](https://github.com/wkhadgar/zview/issues) if you feel like this has some potential!
