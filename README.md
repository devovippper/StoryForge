# StoryForge

Stories forged in your own silicon. StoryForge is a terminal interactive-fiction game powered by local Ollama models.

## Performance modes

- **Low** keeps the original small models and behavior.
- **Medium** targets 8 GB RAM and a GTX 1020 with 2 GB VRAM.
- **High** targets 16 GB RAM and an RTX 2060 with 6 GB VRAM.

Choose a mode with `p` from the main menu or `x`, then `p`, in Settings. Every new scenario is expanded by the narrator into a lived-in world with cities, notable people, player background and starting items, and important local characters. While the world is generated, the TUI shows the current section, elapsed time, and an approximate ETA based on completed sections. Stories start without an assigned quest, mission, or urgent obligation; leads and longer adventures can emerge from meeting people. The Medieval scenario starts you as a poor villager in a small village, with only a few basic belongings. Narration stays with the player's perspective, and after traveling, the next scene and action choices use the new location. Press `p` during a story to talk with people, ask about a trade, or invite someone into your party; press `c` to choose someone and start a conversation. Select **Info** in a character interaction to review known details such as their looks, clothing, occupation, traits, and family. New important characters are generated with specific appearances, outfits, jobs, and personalities. Party members' loyalty can change through meaningful shared events and how you treat their families. Major story turns from social actions are possible, but intentionally uncommon. Medium and High also enable saved story events, character memories, detailed relationship changes, and completed trades. Press `u` during a story to edit places, characters, and memories. The hardware figures are targets; actual speed and memory use depend on Ollama, quantization, context size, and CPU/GPU offload.

When starting a story, choose a character gender and age (15-40), then write a short one-line description. The narrator generates traits from that description and world scenario; traits can influence narration and how people react. Change narration style in Settings with `n`: player's senses, balanced (default), or full cinematic narration. StoryForge is terminal-based, so the character description is confirmed with Enter; standard terminals do not consistently distinguish Shift+Enter.

In Settings, press `m` to switch between **Classic** (stream the narrator as it writes) and **Loading** (show world-specific tips while the narrator writes, then display the finished scene). The selected scene length is a target, not a cutoff; story scenes and world generation can continue as long as needed to finish naturally. Both modes show a live, approximate `tok/s` generation rate; world creation also shows its current section, elapsed time, and approximate ETA. The rate is estimated from streamed text and marked with `~`; Ollama's exact generation rate is available only after a request finishes. In the TUI the rate appears in the status footer; in classic CLI mode it appears in the terminal title while the narrator is writing. Both display modes clear previous terminal output between story scenes; browse the saved-in-memory turn history with the Up/Down arrows and return with Enter or Esc. Press `m` during a story for the ASCII world map, including generated cities, villages, forts, workplaces, and landmarks; road paths and map symbols are drawn by the game. Press `Shift+M` for a generated directory of nearby places and rooms inside your current location, with Exit always listed. The local directory is generated when you first enter each place and saved with the story. The narrator receives recent actions and scenes, the running summary, and persistent character/world details to help it continue established storylines rather than constantly inventing new ones.

Enhanced saves also track a separate world state (time, weather, locations, factions, reputation, and objects), validated world consequences with causal links, active plot threads and foreshadowing, delayed NPC actions, character goals/priorities/arcs/emotions, and confidence-tagged private knowledge. Character memories decay by age and importance, adjusted by relationship status and whether the character wants to remember. Press `d` during a story to inspect state, or `a` to branch into a new timeline save without replacing the original.

## Run

Install and start [Ollama](https://ollama.com/), then run:

```sh
./setup.sh
./storyforge.py
```

The default launch uses a full-screen terminal interface. Use `./storyforge.py --cli` to open the classic line-oriented menu instead. Existing `python3 Storyforge.py` launches still work and also use the TUI by default; add `--cli` to keep the classic interface. `--benchmark` remains a terminal command and does not enter the TUI.

`setup.sh` selects and saves a performance mode. You can change it later with `p` from the main menu or `x`, then `p`, in Settings.

Models selected by a profile that are not installed can be pulled from Settings > Models per role > `p`, then choose Ollama or Hugging Face. The Hugging Face option downloads a mapped Q4_K_M GGUF and imports it into the same Ollama model tag; downloads are cached under StoryForge's data directory and streamed to the configured Ollama host. If Ollama is unreachable, downloads are cached and can be imported by repeating the pull later. Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment if a selected repository requires authentication. This bypasses Ollama's model-library download, not Ollama inference itself; the configured Ollama server must still be reachable for import and play. Benchmark with `python3 Storyforge.py --benchmark`.
