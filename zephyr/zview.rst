.. _external_module_zview:

ZView Runtime Visualizer
########################

Introduction
************

ZView is a system-wide runtime visualizer for Zephyr RTOS that operates via an SWD probe.
It allows developers to take a broader look at their Zephyr application using a non-heavy,
small footprint, Kconfig-only thread stats analyzer.

It achieves a minimal footprint by avoiding on-target processing or UART/Shell output.
Instead, it utilizes the debug probe's ability to read memory via the APB bus without
halting the CPU. By parsing the ELF file, ZView identifies kernel object locations
and performs analysis of stack watermarks and CPU usage.

This module is hosted externally and acts as a TUI (Text User Interface) tool.

Usage with Zephyr
*****************

Prerequisites
=============

To properly analyze your Zephyr app, your ELF binary must be compiled with specific Kconfig
options enabled.

**Required Options:**

.. code-block:: cfg

   CONFIG_INIT_STACKS=y            # Required for stack watermarks
   CONFIG_THREAD_MONITOR=y         # Required for thread discovery
   CONFIG_THREAD_STACK_INFO=y      # Required for thread metadata

**Optional Features:**

.. code-block:: cfg

   CONFIG_THREAD_NAME=y            # Enables thread name display
   CONFIG_THREAD_RUNTIME_STATS=y   # Enables CPU usage tracking
   CONFIG_SYS_HEAP_RUNTIME_STATS=y # Enables heap runtime stats

.. note::
   The ``idle`` thread is implicit and only expresses itself on the used CPU % when available.

Installation
============

You can integrate ZView into your workflow using ``west`` or install it manually via ``pip``.

**West Integration (Recommended)**

Add this snippet to your west manifest:

.. code-block:: yaml

   manifest:
     projects:
       # ...
       - name: zview
         url: https://github.com/wkhadgar/zview
         revision: main
         path: modules/tools/zview
         west-commands: scripts/west-commands.yml

Then run:

.. code-block:: shell

   west update zview
   west zview

**Manual Installation**

.. code-block:: shell

   # From the root of the zview repository
   pip install -e .

   # Or directly through pip
   pip install zview

Running the Tool
================

If running manually from the CLI:

.. code-block:: shell

   zview -e build/zephyr/zephyr.elf -r runner -t runner_mcu_descriptor

**CLI Arguments:**

* ``-e, --elf-file``: Path to the firmware ``.elf`` file.
* ``-r, --runner``: Manually select ``jlink`` or ``pyocd``.
* ``-t, --runner-target``: MCU descriptor name, as used by the chosen runner.
* ``--period``: Update period in seconds (can be float).

Navigation
==========

ZView acts as a TUI. Navigate with **UP** and **DOWN** arrows from the default view.

* **ENTER**: Track CPU usage for a specific thread (hit ENTER again to return).
* **S / I**: Sort the data and Invert the sorting order.
* **H**: Access the **Heap Runtime** visualization.

Reference
*********

* `ZView GitHub Repository <https://github.com/wkhadgar/zview>`_
* `Issue Tracker <https://github.com/wkhadgar/zview/issues>`_
