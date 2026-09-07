"""Executable entry point; divert worker processes before importing Qt."""
import multiprocessing


if __name__ == "__main__":
    multiprocessing.freeze_support()

    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-test":
        from shared.frozen_smoke import run
        sys.exit(run(sys.argv[2]))

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "CyanKualus.NeuroCasting")

    from app import main
    main()
