"""Launch the CoPhaser GUI from anywhere: `python -m cophaser.gui` or `cophaser-gui`."""

import os
import sys


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit(
            "The CoPhaser GUI needs streamlit, which is an optional dependency.\n"
            'Install it with:  pip install "cophaser[gui]"'
        )

    # No [theme] options here: setting any pins Streamlit to light mode.
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = ["streamlit", "run", app_path, *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
