import curses
import sys
from pathlib import Path

from west import log
from west.commands import WestCommand

_zview_base = Path(__file__).parent.parent.parent
sys.path.append(str(_zview_base / "src"))

from backend.z_scraper import JLinkScraper, PyOCDScraper, RunnerConfig, ZScraper
from frontend.zview_tui import tui_run


class ZViewCommand(WestCommand):
    def __init__(self):
        super().__init__(
            'zview',
            'Zephyr RTOS system-wide runtime visualizer via SWD probe',
            'Detailed description of what ZView does.',
        )

        self.available_runners = ["jlink", "pyocd"]

    def do_add_parser(self, parser_adder):
        parser = parser_adder.add_parser(self.name, help=self.help, description=self.description)

        parser.add_argument(
            "-e", "--elf-file", help="Path to the application's .elf firmware file."
        )
        parser.add_argument(
            "-r",
            "--runner",
            choices=self.available_runners,
            default=None,
            help="Runner to start analysis with.",
        )
        parser.add_argument(
            "-t",
            "--runner-target",
            default=None,
            help="Descriptor name for the target MCU on the chosen runner",
        )
        parser.add_argument(
            "--period",
            default=0.10,
            type=float,
            help="Minimum period to update system information.",
        )

        return parser

    def do_run(self, args, unknown):
        del unknown

        build_path = Path('build').resolve()

        if args.elf_file:
            elf_path = Path(args.elf_file)
        else:
            elf_path = build_path / "zephyr" / "zephyr.elf"

            if not elf_path.exists():
                log.die(
                    f"Could not find firmware at '{elf_path}'."
                    " Please specify '-e' or run west build."
                )

        runners_yaml_path = build_path / "zephyr" / "runners.yaml"

        runner = args.runner
        target_mcu = args.runner_target

        if runner and target_mcu:
            pass
        else:
            try:
                runner_config = RunnerConfig(runners_yaml_path)
            except Exception as e:
                log.err(e)
                log.die(
                    "Failed to parse runners.yaml. Please specify runner ('-r') and target ('-t')."
                )

            runner, target_mcu = runner_config.get_config(preferred_runner=runner)
            if runner not in self.available_runners:
                log.wrn(
                    f"'{runner}' is not a valid runner for ZView. Try explicitly passing one of "
                    f"the following: {', '.join(self.available_runners)}"
                )
                log.wrn("Trying to continue with PyOCD as runner...")

        try:
            scraper_cls = PyOCDScraper if runner != "jlink" else JLinkScraper

            with scraper_cls(target_mcu) as meta_scraper:
                z_scraper = ZScraper(meta_scraper, elf_path)
                curses.wrapper(tui_run, z_scraper, args.period)
        except Exception as e:
            log.err(e)
            log.die("Error starting ZView, try 'west zview -h'")
