# StoryForge

Stories forged in your own silicon. StoryForge is a terminal interactive-fiction game powered by local Ollama models.

## Performance modes

- **Low** keeps the original small models and behavior.
- **Medium** targets 8 GB RAM and a GTX 1020 with 2 GB VRAM.
- **High** targets 16 GB RAM and an RTX 2060 with 6 GB VRAM.

Choose a mode with `p` from the main menu or `x`, then `p`, in Settings. Medium and High enable saved story events, character memories, persistent trust/affinity/respect relationships, AI-driven character conversations and accepted trades. Press `u` during a story to edit places, characters, memories, and relationships. The hardware figures are targets; actual speed and memory use depend on Ollama, quantization, context size, and CPU/GPU offload.

Enhanced saves also track a separate world state (time, weather, locations, factions, reputation, and objects), validated world consequences with causal links, active plot threads and foreshadowing, delayed NPC actions, character goals/priorities/arcs/emotions, and confidence-tagged private knowledge. Character memories decay by age and importance, adjusted by relationship status and whether the character wants to remember. Press `d` during a story to inspect state, or `a` to branch into a new timeline save without replacing the original.

## Run

Install and start [Ollama](https://ollama.com/), then run:

```sh
./setup.sh
python3 Storyforge.py
```

`setup.sh` selects and saves a performance mode. You can change it later with `p` from the main menu or `x`, then `p`, in Settings.

Models selected by a profile that are not installed can be pulled from Settings > Models per role > `p`, then choose Ollama or Hugging Face. The Hugging Face option downloads a mapped Q4_K_M GGUF and imports it into the same Ollama model tag; downloads are cached under StoryForge's data directory and streamed to the configured Ollama host. If Ollama is unreachable, downloads are cached and can be imported by repeating the pull later. Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in the environment if a selected repository requires authentication. This bypasses Ollama's model-library download, not Ollama inference itself; the configured Ollama server must still be reachable for import and play. Benchmark with `python3 Storyforge.py --benchmark`.
