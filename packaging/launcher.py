"""Entry point used by PyInstaller."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # the GPU worker processes start the frozen app again
    from simon_enhance.app import main
    sys.exit(main())
