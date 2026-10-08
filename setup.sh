#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1

current_mode="$("$PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); import Storyforge; print(Storyforge.load_config()["performance_mode"])' "$SCRIPT_DIR")"
printf 'StoryForge performance setup\n'
printf '  1) Low     current models\n'
printf '  2) Medium  8 GB RAM, GTX 1020 2 GB VRAM\n'
printf '  3) High    16 GB RAM, RTX 2060 6 GB VRAM\n'
if ! IFS= read -r -p "Select mode [current: $current_mode]: " choice; then
    printf '\nSetup cancelled; existing configuration was not changed.\n'
    exit 1
fi
case "${choice,,}" in
    "") mode="$current_mode" ;;
    1|l|low) mode="low" ;;
    2|m|medium) mode="medium" ;;
    3|h|high) mode="high" ;;
    *) printf 'Choose 1/Low, 2/Medium, or 3/High.\n' >&2; exit 2 ;;
esac

"$PYTHON" - "$SCRIPT_DIR" "$mode" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1])))
import Storyforge as storyforge

cfg = storyforge.load_config()
mode = sys.argv[2]
storyforge.apply_performance_mode(cfg, mode)
storyforge.save_config(cfg)
saved = storyforge.load_config()
if (saved["performance_mode"] != mode or saved["models"] != cfg["models"]
        or saved["num_ctx"] != cfg["num_ctx"]):
    raise SystemExit("Could not save the selected mode to %s" % storyforge.CONFIG_FILE)

profile = storyforge.PERFORMANCE_MODES[mode]
print("Selected %s mode (%s)." % (profile["label"], profile["ram"]))
print("Config saved to %s" % storyforge.CONFIG_FILE)
print("Models not installed yet can be pulled from StoryForge Settings > Models per role > p.")
PY
