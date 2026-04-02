from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.toy_experiment import main

if __name__ == "__main__":
    main()
