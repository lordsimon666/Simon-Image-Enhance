"""Entry point used by PyInstaller."""
import sys

from simon_enhance.app import main

if __name__ == "__main__":
    sys.exit(main())
