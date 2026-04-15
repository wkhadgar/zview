# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from frontend.tui.views.base import Any, BaseStateView, SpecialCode, ZViewState, ZViewTUIAttributes


class FatalErrorView(BaseStateView):
    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()

        self._render_frame(stdscr, "Quit: q | Reconnect: r ", height, width)

        stdscr.attron(self._error_attr)
        msg_lines = self.controller.status_message.split('\n')
        start_y = (height // 2) - (len(msg_lines) // 2)

        for i, line in enumerate(msg_lines):
            if 0 <= start_y + i < height - 2:
                clean_line = line[: width - 2]
                x_pos = max(0, (width // 2) - (len(clean_line) // 2))
                stdscr.addstr(start_y + i, x_pos, clean_line)

        stdscr.attroff(self._error_attr)
        stdscr.refresh()

    def handle_input(self, key: int) -> ZViewState | None:
        if key == SpecialCode.QUIT:
            self.controller.running = False

        elif key == SpecialCode.RECONNECT:
            self.controller.attempt_reconnect()

        return None
