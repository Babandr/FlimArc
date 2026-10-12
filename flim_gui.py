#!/usr/bin/env python3
"""
Flimera FLIM Analysis -- GUI entry point.

Run with:
    python flim_gui.py

This file only launches the app; all logic lives in gui/mainwindow.py,
which in turn drives whatever plugins analyses/_registry.py discovers.
"""

from gui.mainwindow import FlimMasterApp


def main():
    app = FlimMasterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
