#!/usr/bin/env python3
"""StoryForge - interactive stories driven by tiny local Ollama models.

Each stage of a turn uses its assigned model. Select Low, Medium, or High from the main menu
or x > p; edit individual models from x > 1.

  Memory    nomic-embed-text    recalls earlier events relevant to what you just did
  Narrator  qwen2.5:0.5b        writes the next scene (streamed); expands custom worlds
  Analyst   gemma3:270m         mood / danger / characters, rolling story summary
  Choices   qwen2.5-coder:0.5b  the next actions, as strict JSON
  Tracker   functiongemma:270m  tool calls that update location, items, health, goal
  Router    smollm2:135m        classifies typed input, quick "look around" replies

Low uses the original models. Medium targets 8 GB RAM / 2 GB GTX 1020 VRAM; High targets
16 GB RAM / 6 GB RTX 2060 VRAM. Medium and High add persistent events, character memories,
relationships, AI-driven dialogue and trades, plus the in-game u world editor. Models run
one after another, with profile-specific limits on how many stay loaded. Pure ASCII output, so it also looks right on a
Linux text console or a small USB display. Needs Python 3.8+ and a running Ollama.

  python3 storyforge.py                 play
  python3 storyforge.py --benchmark     benchmark the models and exit
  python3 storyforge.py --host 192.168.1.20:11434
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

VERSION = "1.0"

# --------------------------------------------------------------------------- paths
_home = os.environ.get("STORYFORGE_HOME")
if _home:
    CONFIG_DIR = DATA_DIR = Path(_home)
else:
    CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "storyforge"
    DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "storyforge"
CONFIG_FILE = CONFIG_DIR / "config.json"
SAVE_DIR = DATA_DIR / "saves"
EXPORT_DIR = DATA_DIR / "exports"
BENCH_FILE = DATA_DIR / "benchmark.json"
HF_MODEL_DIR = DATA_DIR / "huggingface" / "models"

# --------------------------------------------------------------------------- settings
# role -> (default model, label, short description)
ROLES = {
    "narrator": ("qwen2.5:0.5b", "Narrator", "story prose"),
    "choices": ("qwen2.5-coder:0.5b", "Choices", "JSON next-actions"),
    "analyst": ("gemma3:270m", "Analyst", "summary, mood, cast"),
    "router": ("smollm2:135m", "Router", "typed input, quick replies"),
    "tools": ("functiongemma:270m", "Tracker", "state tool calls"),
    "memory": ("nomic-embed-text", "Memory", "recalls older events"),
}
PERFORMANCE_MODES = {
    "low": {"label": "Low", "ram": "current models", "max_loaded": None, "num_ctx": 2048,
            "models": {k: v[0] for k, v in ROLES.items()}},
    "medium": {"label": "Medium", "ram": "8 GB RAM, GTX 1020 2 GB VRAM",
               "max_loaded": 1, "num_ctx": 4096,
               "models": {"narrator": "qwen2.5:1.5b", "choices": "qwen2.5-coder:1.5b",
                          "analyst": "gemma3:1b", "router": "gemma3:1b",
                          "tools": "functiongemma:270m", "memory": "nomic-embed-text"}},
    "high": {"label": "High", "ram": "16 GB RAM, RTX 2060 6 GB VRAM",
             "max_loaded": 2, "num_ctx": 8192,
             "models": {"narrator": "qwen2.5:3b", "choices": "qwen2.5-coder:3b",
                        "analyst": "gemma3:4b", "router": "qwen2.5:3b",
                        "tools": "functiongemma:270m", "memory": "nomic-embed-text"}},
}
HF_GGUF_SOURCES = {
    "qwen2.5:0.5b": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    "qwen2.5:1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "qwen2.5:3b": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "qwen2.5-coder:0.5b": ("Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
                            "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-coder:1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
                            "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
    "qwen2.5-coder:3b": ("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
                         "qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
    "gemma3:270m": ("lmstudio-community/gemma-3-270m-it-GGUF", "gemma-3-270m-it-Q4_K_M.gguf"),
    "gemma3:1b": ("lmstudio-community/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"),
    "gemma3:4b": ("lmstudio-community/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    "smollm2:135m": ("lmstudio-community/SmolLM2-135M-Instruct-GGUF",
                     "SmolLM2-135M-Instruct-Q4_K_M.gguf"),
    "functiongemma:270m": ("unsloth/functiongemma-270m-it-GGUF", "functiongemma-270m-it-Q4_K_M.gguf"),
    "nomic-embed-text": ("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
}
LENGTHS = {"short": 60, "medium": 100, "long": 160}
KEEP_ALIVES = ["1m", "5m", "30m"]

DEFAULTS = {
    "host": os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"),
    "models": {k: v[0] for k, v in ROLES.items()},
    "performance_mode": "low",
    "max_loaded": None,      # None = auto from RAM
    "keep_alive": "5m",
    "length": "medium",
    "temperature": 0.8,
    "choices": 3,
    "recall": 3,             # 0 = off
    "tracking": True,
    "stream": True,
    "timing": True,
    "num_ctx": 2048,
}


def load_config():
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        with open(CONFIG_FILE) as f:
            user = json.load(f)
        for k, v in user.items():
            if k == "models" and isinstance(v, dict):
                cfg["models"].update({r: m for r, m in v.items() if r in ROLES and isinstance(m, str)})
            elif k in cfg:
                cfg[k] = v
    except (OSError, ValueError):
        pass
    if cfg["length"] not in LENGTHS:
        cfg["length"] = "medium"
    if cfg.get("performance_mode") not in PERFORMANCE_MODES:
        cfg["performance_mode"] = "low"
    return cfg


def save_config(cfg):
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass


def auto_max_loaded():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    gb = int(line.split()[1]) / 1048576
                    return 1 if gb < 2.6 else 2 if gb < 4.6 else 3
    except (OSError, ValueError, IndexError):
        pass
    return 2


def effective_max_loaded(cfg):
    return cfg["max_loaded"] or auto_max_loaded()


def apply_performance_mode(cfg, mode):
    profile = PERFORMANCE_MODES[mode]
    cfg["performance_mode"] = mode
    cfg["models"] = dict(profile["models"])
    cfg["max_loaded"] = profile["max_loaded"]
    cfg["num_ctx"] = profile["num_ctx"]


# --------------------------------------------------------------------------- story data
PRESETS = [
    dict(name="Medieval", blurb="gritty kingdoms, politics, a hard winter",
         tone="gritty and low-magic",
         setting="A feudal kingdom of stone keeps, muddy roads and uneasy alliances. A hard winter is closing in.",
         role="a young squire", location="the gates of Castle Harrowmere",
         objective="deliver a sealed letter to the Lord of Harrowmere",
         items=["sealed letter", "worn dagger", "heel of bread"]),
    dict(name="Fantasy", blurb="wizards, ruins and waking magic",
         tone="wondrous and adventurous",
         setting="A land of floating isles, talking beasts and forgotten spells, where old magic is waking up.",
         role="an apprentice mage", location="the Whispering Library",
         objective="find the stolen spellbook of your master",
         items=["oak staff", "glowing pebble", "tattered notes"]),
    dict(name="Sci-fi", blurb="derelict stations and lonely stars",
         tone="tense, lonely and curious",
         setting="The year 2412. A drifting mining station orbits a dying star, and its crew has vanished.",
         role="a salvage pilot", location="the airlock of Kepler Station",
         objective="restore power and learn what happened to the crew",
         items=["plasma torch", "comm badge", "ration pack"]),
    dict(name="Cyberpunk", blurb="neon, corporations, dangerous data",
         tone="fast, gritty and neon-lit",
         setting="Rain-soaked megacity streets ruled by corporations, hackers and street gangs.",
         role="a street courier", location="a noodle bar in Neon Alley",
         objective="deliver a data chip before dawn without being caught",
         items=["data chip", "cracked phone", "stun baton"]),
    dict(name="Horror", blurb="creeping dread in an empty place",
         tone="slow-burn, creepy and atmospheric",
         setting="An abandoned seaside asylum where the lights flicker and the doors do not stay shut.",
         role="a night-shift caretaker", location="the dark main hall of Blackmoor Asylum",
         objective="find the source of the whispering and get out alive",
         items=["flashlight", "ring of keys"]),
    dict(name="Noir", blurb="rainy streets, lies, a missing person",
         tone="hard-boiled and cynical",
         setting="A rain-slicked 1940s city of smoky bars, crooked cops and secrets.",
         role="a tired private detective", location="your cluttered office",
         objective="find out who sent the mysterious client and what she wants",
         items=["notebook", "revolver", "hip flask"]),
    dict(name="Post-apoc", blurb="scavenging in a ruined world",
         tone="bleak but hopeful",
         setting="Decades after the collapse: ruined highways, rusted towns and scattered settlements.",
         role="a lone scavenger", location="a collapsed highway overpass",
         objective="find clean water for your settlement",
         items=["crowbar", "water flask", "gas mask"]),
    dict(name="Pirates", blurb="storms, treasure and mutiny",
         tone="swashbuckling and humorous",
         setting="The Sunken Sea, full of storms, cursed islands and rival crews.",
         role="a stowaway", location="the deck of the Salt Maiden",
         objective="uncover the secret of the captain's map",
         items=["rusty cutlass", "bit of rope", "lucky coin"]),
]

SETUP_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "tone": {"type": "string"}, "setting": {"type": "string"},
        "role": {"type": "string"}, "location": {"type": "string"}, "objective": {"type": "string"},
        "items": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "tone", "setting", "role", "location", "objective", "items"],
}
MOODS = ["calm", "tense", "eerie", "joyful", "sad", "action"]
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "mood": {"type": "string", "enum": MOODS},
        "danger": {"type": "integer", "enum": [0, 1, 2, 3]},
        "characters": {"type": "array", "items": {"type": "string"}},
        "location": {"type": "string"},
    },
    "required": ["mood", "danger", "characters", "location"],
}
ENHANCED_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "mood": {"type": "string", "enum": MOODS},
        "danger": {"type": "integer", "enum": [0, 1, 2, 3]},
        "characters": {"type": "array", "items": {"type": "string"}},
        "location": {"type": "string"},
        "event": {"type": "string"},
        "character_notes": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "memory": {"type": "string"},
            "wants_remember": {"type": "boolean"}, "importance": {"type": "integer", "minimum": 0, "maximum": 100},
            "emotional_strength": {"type": "integer", "minimum": 0, "maximum": 100},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "source": {"type": "string"}, "valence": {"type": "integer", "minimum": -100, "maximum": 100}},
            "required": ["name", "memory", "wants_remember", "importance", "emotional_strength",
                         "confidence", "source", "valence"]}},
        "relationships": {"type": "array", "items": {"type": "object", "properties": {
            "character": {"type": "string"}, "trust_change": {"type": "integer"},
            "affinity_change": {"type": "integer"}, "respect_change": {"type": "integer"},
            "note": {"type": "string"}}, "required": ["character", "trust_change",
                                                            "affinity_change", "respect_change", "note"]}},
        "character_updates": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "goal": {"type": "string"},
            "motivations": {"type": "array", "items": {"type": "string"}},
            "priorities": {"type": "array", "items": {"type": "string"}},
            "knowledge": {"type": "array", "items": {"type": "object", "properties": {
                "fact": {"type": "string"}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "source": {"type": "string"}}, "required": ["fact", "confidence", "source"]}},
            "anger_change": {"type": "integer"}, "fear_change": {"type": "integer"},
            "stress_change": {"type": "integer"}, "confidence_change": {"type": "integer"},
            "arc": {"type": "string"}, "arc_stage": {"type": "string"}},
            "required": ["name", "goal", "motivations", "priorities", "knowledge", "anger_change", "fear_change",
                         "stress_change", "confidence_change", "arc", "arc_stage"]}},
        "world_changes": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["world", "location", "faction", "reputation", "object",
                "fact_add", "fact_resolve", "thread_add", "thread_resolve", "foreshadow_add", "foreshadow_reference",
                "foreshadow_resolve", "foreshadow_abandon", "schedule"]},
            "target": {"type": "string"}, "field": {"type": "string"},
            "value": {"type": "string"}, "delta": {"type": "integer"},
            "reason": {"type": "string"}, "due_minutes": {"type": "integer", "minimum": 0}},
            "required": ["kind", "target", "field", "value", "delta", "reason", "due_minutes"]}},
        "time_advance_minutes": {"type": "integer", "minimum": 0, "maximum": 1440},
        "scene_objective": {"type": "string"}, "scene_stakes": {"type": "string"},
        "player_knowledge": {"type": "array", "items": {"type": "object", "properties": {
            "fact": {"type": "string"}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "source": {"type": "string"}}, "required": ["fact", "confidence", "source"]}},
    },
    "required": ["mood", "danger", "characters", "location", "event", "character_notes", "relationships",
                 "character_updates", "world_changes", "time_advance_minutes", "scene_objective", "scene_stakes",
                 "player_knowledge"],
}


def _tool(name, desc, prop, ptype="string", pdesc=""):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {prop: {"type": ptype, "description": pdesc}},
                       "required": [prop]}}}


TOOLS = [
    _tool("add_item", "The player picks up or receives an item", "item", "string", "item name"),
    _tool("remove_item", "The player loses, uses up or gives away an item", "item", "string", "item name"),
    _tool("change_location", "The player moves to a different place", "place", "string", "new place"),
    _tool("adjust_health", "The player is hurt (negative) or healed (positive)", "amount", "integer", "health change"),
    _tool("set_objective", "The player's goal changes", "objective", "string", "new goal"),
    {"type": "function", "function": {"name": "trade_items",
     "description": "Complete an agreed trade described in the passage; call only after the other character accepts.",
     "parameters": {"type": "object", "properties": {
         "character": {"type": "string"}, "give_item": {"type": "string"},
         "receive_item": {"type": "string"}},
         "required": ["character", "give_item", "receive_item"]}}},
]

KEYWORDS = {
    "look": {"look", "l", "look around", "where am i", "what do i see"},
    "inventory": {"i", "inv", "inventory", "items", "status", "what do i have"},
    "help": {"help", "?", "h"},
    "quit": {"quit", "exit", "menu"},
}
ROUTE_PROMPT = ("You sort messages typed by a player of a text adventure. Answer with exactly one word: "
                "action (the player does something), talk (the player speaks to someone), "
                "trade (the player proposes an exchange), look (the player wants to see their surroundings) "
                "or inventory (the player asks what they carry).\nMessage: ")
FALLBACK_CHOICES = ["Look around carefully", "Press onward", "Wait and listen", "Turn back"]


# --------------------------------------------------------------------------- terminal helpers
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM", "") != "dumb"


def paint(code, s):
    return "\033[%sm%s\033[0m" % (code, s) if COLOR else s


def dim(s): return paint("2", s)
def bold(s): return paint("1", s)
def green(s): return paint("32", s)
def red(s): return paint("31", s)
def yellow(s): return paint("33", s)


def term_width():
    return max(40, min(100, shutil.get_terminal_size((80, 24)).columns - 1))


def say(s=""):
    print(s)
    sys.stdout.flush()


def rule(title="", ch="="):
    w = term_width()
    if title:
        title = " %s " % title
    return (ch * 3 + title).ljust(w, ch)[:w]


def getkey():
    """One keypress (lower-cased). Falls back to line input when stdin is not a terminal."""
    sys.stdout.flush()
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        line = line.strip()
        return line[:1].lower() if line else "\n"
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            return "esc"
        if ch == "\x03":
            raise KeyboardInterrupt
        return "\n" if ch == "\r" else ch.lower()
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = os.read(fd, 1).decode("utf-8", "ignore")
        if ch == "\x1b":                       # swallow the rest of an arrow-key sequence
            while select.select([fd], [], [], 0.03)[0]:
                os.read(fd, 1)
            return "esc"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch == "\x04":
        raise EOFError
    return "\n" if ch in ("\r", "\n") else ch.lower()


try:                                           # line editing for typed input
    import readline  # noqa: F401
except ImportError:
    pass


def ask(prompt="> "):
    return input(prompt).strip()


class Status:
    """Transient 'working...' line with a seconds counter (only on a real terminal)."""

    def __init__(self, text):
        self.text, self.t0 = text, time.time()
        self.ev, self.thread = threading.Event(), None
        if sys.stdout.isatty():
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        while True:
            sys.stdout.write("\r\033[K" + dim("  ... %s %ds" % (self.text, int(time.time() - self.t0))))
            sys.stdout.flush()
            if self.ev.wait(1.0):
                break

    def stop(self):
        if self.ev.is_set():
            return
        self.ev.set()
        if self.thread:
            self.thread.join()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


class Wrap:
    """Word-wraps streamed text to the terminal width."""

    def __init__(self, width, indent="  "):
        self.width, self.indent = width, indent
        self.col, self.word, self.blank = 0, "", True

    def _raw(self, s):
        sys.stdout.write(s)
        sys.stdout.flush()

    def _flush(self):
        if not self.word:
            return
        if self.col == 0:
            self._raw(self.indent)
            self.col = len(self.indent)
        elif self.col + 1 + len(self.word) > self.width:
            self._raw("\n" + self.indent)
            self.col = len(self.indent)
        else:
            self._raw(" ")
            self.col += 1
        self._raw(self.word)
        self.col += len(self.word)
        self.word, self.blank = "", False

    def feed(self, s):
        for ch in s:
            if ch == "\n":
                self._flush()
                if self.col:
                    self._raw("\n")
                    self.col = 0
                elif not self.blank:
                    self._raw("\n")
                    self.blank = True
            elif ch.isspace():
                self._flush()
            else:
                self.word += ch

    def close(self):
        self._flush()
        if self.col:
            self._raw("\n")
            self.col = 0


# --------------------------------------------------------------------------- text helpers
def norm(name):
    n = name.strip()
    return n if ":" in n else n + ":latest"


def brief(text, n):
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut + "..."


def tidy(text):
    t = text.strip()
    t = re.sub(r"^(?:narrator|story|assistant)\s*:\s*", "", t, flags=re.I)
    m = re.search(r"\n\s*(?:choices|options|what (?:do|will) you do)\b", t, re.I)
    if m:
        t = t[:m.start()]
    m = re.search(r"\n\s*1[.)]\s+\S", t)
    if m:
        t = t[:m.start()]
    t = re.sub(r"[*_#`]+", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if t and t[-1] not in ".!?\"'”)":
        cut = max(t.rfind(". "), t.rfind("! "), t.rfind("? "), t.rfind(".\n"))
        if cut > len(t) * 0.5:
            t = t[:cut + 1]
    return t


def sstr(v, n):
    return str(v).strip()[:n] if isinstance(v, (str, int, float)) else ""


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def clean_choices(raw, n):
    out = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, str):
            continue
        c = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", c).strip(" \"'.")
        if 2 <= len(c) <= 70 and c.lower() not in [o.lower() for o in out]:
            out.append(c[0].upper() + c[1:])
    return out[:n]


def parse_intent(out):
    m = re.search(r"\b(action|talk|trade|look|inventory)\b", out.lower())
    return m.group(1) if m else "action"


def cycle(options, cur):
    i = options.index(cur) if cur in options else -1
    return options[(i + 1) % len(options)]


def onoff(v):
    return "on" if v else "off"


# --------------------------------------------------------------------------- Ollama client
class OllamaError(Exception):
    def __init__(self, msg, status=0):
        super().__init__(msg)
        self.status = status          # 0 = could not connect / timeout


class Unavailable(Exception):
    pass


def norm_host(host):
    h = host.strip().rstrip("/")
    if "://" not in h:
        h = "http://" + h
    rest = h.split("://", 1)[1]
    if ":" not in rest:
        h += ":11434"
    return h.replace("//0.0.0.0", "//127.0.0.1")


def stats_of(f, wall):
    ns = 1e9
    ev, evd = f.get("eval_count", 0), f.get("eval_duration", 0)
    pc, pd = f.get("prompt_eval_count", 0), f.get("prompt_eval_duration", 0)
    return {"load": f.get("load_duration", 0) / ns,
            "gen_tps": ev / (evd / ns) if evd else 0.0,
            "prompt_tps": pc / (pd / ns) if pd else 0.0,
            "tokens": ev, "wall": wall}


class Ollama:
    def __init__(self, host, timeout=600):
        self.base, self.timeout = norm_host(host), timeout

    def _req(self, path, payload=None, method="POST"):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                msg = json.loads(body).get("error", body)
            except (ValueError, AttributeError):
                msg = body
            raise OllamaError(str(msg).strip() or "HTTP %d" % e.code, e.code) from None
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError("cannot reach Ollama at %s (%s)" % (self.base, getattr(e, "reason", e))) from None

    def get(self, path):
        with self._req(path, method="GET") as r:
            return json.loads(r.read().decode())

    def version(self):
        return self.get("/api/version").get("version", "?")

    def tags(self):
        return [m["name"] for m in self.get("/api/tags").get("models", [])]

    def ps(self):
        return self.get("/api/ps").get("models", [])

    def chat(self, model, messages, fmt=None, tools=None, options=None, keep_alive=None, on_token=None):
        payload = {"model": model, "messages": messages, "stream": on_token is not None}
        if fmt is not None:
            payload["format"] = fmt
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        try:
            return self._chat(payload, on_token)
        except OllamaError as e:
            if e.status == 400 and isinstance(fmt, dict):      # older server: no JSON-schema support
                payload["format"] = "json"
                return self._chat(payload, on_token)
            raise

    def _chat(self, payload, on_token):
        t0 = time.time()
        content, calls, final = [], [], {}
        with self._req("/api/chat", payload) as r:
            if on_token is None:
                chunks = [json.loads(r.read().decode())]
            else:
                chunks = (json.loads(line) for line in r if line.strip())
            for obj in chunks:
                if "error" in obj:
                    raise OllamaError(str(obj["error"]), 500)
                msg = obj.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    content.append(piece)
                    if on_token:
                        on_token(piece)
                calls.extend(msg.get("tool_calls") or [])
                if obj.get("done"):
                    final = obj
        return {"content": "".join(content), "tool_calls": calls, "stats": stats_of(final, time.time() - t0)}

    def embed(self, model, texts, keep_alive=None):
        payload = {"model": model, "input": texts}
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        t0 = time.time()
        obj = {}
        try:
            with self._req("/api/embed", payload) as r:
                obj = json.loads(r.read().decode())
            vecs = obj.get("embeddings") or []
        except OllamaError as e:
            if e.status != 404:
                raise
            vecs = []                                           # older servers
            for t in texts:
                with self._req("/api/embeddings", {"model": model, "prompt": t}) as r:
                    vecs.append(json.loads(r.read().decode()).get("embedding") or [])
        if len(vecs) != len(texts) or not all(vecs):
            raise OllamaError("the embedding model returned no vectors", 500)
        return vecs, stats_of(obj, time.time() - t0)

    def unload(self, model):
        try:
            with self._req("/api/generate", {"model": model, "keep_alive": 0}) as r:
                r.read()
        except OllamaError:
            try:
                with self._req("/api/embed", {"model": model, "input": "", "keep_alive": 0}) as r:
                    r.read()
            except OllamaError:
                pass

    def pull(self, model, on_progress):
        with self._req("/api/pull", {"model": model, "stream": True}) as r:
            for line in r:
                if line.strip():
                    o = json.loads(line)
                    if "error" in o:
                        raise OllamaError(str(o["error"]), 500)
                    on_progress(o)

    def has_blob(self, digest):
        request = urllib.request.Request(self.base + "/api/blobs/" + digest, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout):
                return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            raise OllamaError("Ollama blob check failed (HTTP %d)" % error.code, error.code) from None
        except (urllib.error.URLError, OSError) as error:
            raise OllamaError("cannot reach Ollama at %s (%s)" % (
                self.base, getattr(error, "reason", error))) from None

    def push_blob(self, path, on_progress):
        size = os.path.getsize(path)
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            completed = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                completed += len(chunk)
                on_progress({"status": "Checking GGUF digest", "completed": completed, "total": size})
        digest = "sha256:" + digest.hexdigest()
        if self.has_blob(digest):
            on_progress({"status": "GGUF blob already exists in Ollama", "completed": size, "total": size})
            return digest

        def upload_chunks():
            completed = 0
            with open(path, "rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    completed += len(chunk)
                    on_progress({"status": "Uploading GGUF to Ollama", "completed": completed, "total": size})
                    yield chunk

        request = urllib.request.Request(self.base + "/api/blobs/" + digest, data=upload_chunks(), method="POST",
                                         headers={"Content-Type": "application/octet-stream",
                                                  "Content-Length": str(size)})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
        except urllib.error.HTTPError as error:
            raise OllamaError("Ollama blob upload failed (HTTP %d)" % error.code, error.code) from None
        except (urllib.error.URLError, OSError) as error:
            raise OllamaError("Ollama blob upload failed (%s)" % getattr(error, "reason", error)) from None
        return digest

    def create(self, model, files, on_progress):
        with self._req("/api/create", {"model": model, "files": files, "stream": True}) as r:
            for line in r:
                if line.strip():
                    obj = json.loads(line)
                    if "error" in obj:
                        raise OllamaError(str(obj["error"]), 500)
                    on_progress(obj)


class Scheduler:
    """Keeps at most max_loaded models in memory by unloading the least recently used."""

    def __init__(self, client, cfg):
        self.client, self.cfg = client, cfg

    def prepare(self, model):
        limit = effective_max_loaded(self.cfg)
        try:
            loaded = self.client.ps()
        except OllamaError:
            return
        names = [norm(m.get("name") or m.get("model") or "") for m in loaded]
        if norm(model) in names:
            return
        order = sorted(loaded, key=lambda m: m.get("expires_at", ""))
        while order and len(order) >= limit:
            victim = order.pop(0)
            self.client.unload(victim.get("name") or victim.get("model"))


class AI:
    """Role-based facade over the client: model lookup, fallbacks, scheduling, timing."""
    FALLBACK = {"choices": "narrator", "analyst": "narrator"}

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = Ollama(cfg["host"])
        self.sched = Scheduler(self.client, cfg)
        self.installed, self.disabled, self.timing = set(), set(), {}
        self.online, self.version = False, "?"

    def reconnect(self):
        self.client = Ollama(self.cfg["host"])
        self.sched.client = self.client
        self.disabled.clear()
        try:
            self.installed = {norm(n) for n in self.client.tags()}
            self.version = self.client.version()
            self.online = True
        except OllamaError:
            self.installed, self.online = set(), False
            raise

    def usable(self, model, role):
        return norm(model) in self.installed and role not in self.disabled

    def has(self, role):
        return self.usable(self.cfg["models"][role], role)

    def model_for(self, role):
        if self.has(role):
            return self.cfg["models"][role]
        fb = self.FALLBACK.get(role)
        return self.model_for(fb) if fb else None

    def missing(self):
        return [(r, m) for r, m in self.cfg["models"].items() if norm(m) not in self.installed]

    def chat(self, role, messages, label, stream=None, **kw):
        model = self.model_for(role)
        if model is None:
            raise Unavailable(role)
        self.sched.prepare(model)
        options = {"num_ctx": self.cfg["num_ctx"]}
        options.update(kw.pop("options", {}))
        t0 = time.time()
        st = Status("%s [%s]" % (label, model))

        def on_token(piece):
            st.stop()
            stream(piece)

        try:
            res = self.client.chat(model, messages, options=options, keep_alive=self.cfg["keep_alive"],
                                   on_token=on_token if stream else None, **kw)
        except OllamaError as e:
            if e.status == 404:
                self.disabled.add(role)
                raise Unavailable(role)
            raise
        finally:
            st.stop()
        self.timing[label] = self.timing.get(label, 0.0) + time.time() - t0
        return res

    def embed(self, texts, label="recall"):
        model = self.cfg["models"]["memory"]
        if not self.usable(model, "memory"):
            raise Unavailable("memory")
        self.sched.prepare(model)
        t0 = time.time()
        st = Status("%s [%s]" % (label, model))
        try:
            vecs, _ = self.client.embed(model, texts, keep_alive=self.cfg["keep_alive"])
        except OllamaError as e:
            if e.status in (400, 404):
                self.disabled.add("memory")
                raise Unavailable("memory")
            raise
        finally:
            st.stop()
        self.timing[label] = self.timing.get(label, 0.0) + time.time() - t0
        return vecs


# --------------------------------------------------------------------------- the game
def default_world_state(location):
    return {"date": "Day 1", "time_minutes": 480, "weather": "unknown", "political_state": "unknown",
            "alert_level": 0, "locations": {location: {"condition": "intact", "owner": "",
                "details": "", "known_events": [], "guards": 0}}, "factions": {}, "reputation": {},
            "objects": {}, "facts": [],
            "threads": [], "foreshadowing": [], "pending_events": [], "causality": [],
            "world_events": [], "scene": {"objective": "", "stakes": "", "participants": []},
            "timeline": {"name": "A", "parent": None, "parent_save_id": None, "branch_count": 0}}


def bounded_int(value, default=0, low=-100, high=100):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def normalize_memory(memory, turn=0):
    if isinstance(memory, str):
        memory = {"text": memory}
    if not isinstance(memory, dict):
        return None
    text = sstr(memory.get("text", memory.get("memory", "")), 180)
    if not text:
        return None
    wants_remember = memory.get("wants_remember", True)
    if not isinstance(wants_remember, bool):
        wants_remember = str(wants_remember).lower() not in ("false", "no", "0", "")
    return {"text": text, "turn": bounded_int(memory.get("turn", turn), turn, 0, 1000000),
            "importance": bounded_int(memory.get("importance", 50), 50, 0, 100),
            "emotional_strength": bounded_int(memory.get("emotional_strength", 40), 40, 0, 100),
            "confidence": bounded_int(memory.get("confidence", 80), 80, 0, 100),
            "source": sstr(memory.get("source", "observed"), 60),
            "wants_remember": wants_remember,
            "valence": bounded_int(memory.get("valence", 0), 0, -100, 100)}


def memory_retention_score(memory, relationship, current_turn):
    memory = normalize_memory(memory)
    if not memory:
        return 0.0
    age = max(0, current_turn - memory["turn"])
    relation = sum(relationship.get(k, 0) for k in ("trust", "affinity", "respect")) / 3.0
    significance = (0.2 + 0.35 * memory["importance"] / 100.0
                    + 0.2 * memory["emotional_strength"] / 100.0
                    + 0.15 * memory["confidence"] / 100.0)
    desire = 1.35 if memory["wants_remember"] else 0.55
    relationship_fit = max(0.5, min(1.5, 1.0 + relation * memory["valence"] / 10000.0))
    return significance * desire * relationship_fit / (1.0 + age / 20.0)


def new_state(p, mode="low"):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return {"id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4],
            "preset": p["name"], "tone": p["tone"], "setting": p["setting"], "role": p["role"],
            "location": p["location"], "objective": p["objective"], "items": list(p["items"]),
            "hp": 10, "max_hp": 10, "characters": [], "summary": "", "summarized_upto": 0,
            "turns": [], "memories": [], "emb_model": None, "choices": [], "over": False,
            "performance_mode": mode, "places": [p["location"]], "events": [],
            "character_notes": {}, "relationships": {}, "character_items": {},
            "world_state": default_world_state(p["location"]), "character_goals": {},
            "character_motivations": {}, "character_priorities": {}, "character_knowledge": {},
            "player_knowledge": [], "character_emotions": {},
            "character_arcs": {}, "relationship_history": [], "memory_archive": [],
            "created": now, "updated": now}


class Game:
    def __init__(self, app, state):
        self.app, self.ai, self.cfg, self.s = app, app.ai, app.cfg, state

    def enhanced(self):
        return self.s.get("performance_mode", self.cfg.get("performance_mode", "low")) in ("medium", "high")

    def character_key(self, name):
        return next((c for c in self.s.get("characters", []) if c.lower() == name.lower()), name)

    def clock_label(self):
        world = self.s.get("world_state", {})
        minutes = max(0, int(world.get("time_minutes", 480)))
        return "%s, %02d:%02d" % (world.get("date") or "Day 1", (minutes // 60) % 24, minutes % 60)

    def character_memories(self, name, limit=3):
        s = self.s
        relation = s.get("relationships", {}).get(name, {})
        memories = [normalize_memory(m) for m in s.get("character_notes", {}).get(name, [])]
        memories = [m for m in memories if m]
        memories.sort(key=lambda m: memory_retention_score(m, relation, len(s.get("turns", []))), reverse=True)
        return memories[:limit]

    def world_context(self):
        world = self.s.get("world_state", {})
        snapshot = {
            "time": self.clock_label(), "weather": world.get("weather"),
            "political_state": world.get("political_state"), "alert_level": world.get("alert_level"),
            "location": world.get("locations", {}).get(self.s["location"], {}),
            "factions": world.get("factions", {}), "reputation": world.get("reputation", {}),
            "objects": world.get("objects", {}),
            "recent_consequences": world.get("world_events", [])[-8:],
            "verified_world_facts": world.get("facts", [])[-12:],
            "current_scene": world.get("scene", {}),
            "active_threads": [t for t in world.get("threads", []) if t.get("status") == "active"][-8:],
            "foreshadowing": [f for f in world.get("foreshadowing", []) if f.get("status") == "active"][-6:],
            "pending_events": world.get("pending_events", [])[:6],
        }
        return json.dumps(snapshot, ensure_ascii=True)[:3000]

    # ---- helpers
    def state_line(self):
        s = self.s
        return "location: %s; time: %s; health: %d/%d; carrying: %s; goal: %s" % (
            s["location"], self.clock_label(), s["hp"], s["max_hp"],
            ", ".join(s["items"]) or "nothing", s["objective"])

    def guard(self, fn, *args, role=None):
        """Run an optional pipeline stage; a failure skips the stage instead of the turn."""
        try:
            return fn(*args)
        except Unavailable:
            return None
        except OllamaError as e:
            if e.status == 0:
                raise
            if role and e.status == 400:
                self.ai.disabled.add(role)
            say(dim("  (%s skipped: %s)" % (fn.__name__, str(e)[:60])))
        except (ValueError, KeyError, TypeError, AttributeError):
            say(dim("  (%s skipped: unusable model output)" % fn.__name__))
        return None

    # ---- pipeline stages
    def recall(self, query):
        s = self.s
        if s["emb_model"] != self.cfg["models"]["memory"]:       # vectors from another model are useless
            for m in s["memories"]:
                m["vec"] = None
            s["emb_model"] = self.cfg["models"]["memory"]
        old = s["memories"][:-2]                                 # the last two scenes are in the prompt anyway
        if not old:
            return []
        need = [m for m in old if not m.get("vec")]
        vecs = self.ai.embed([query] + [m["text"] for m in need])
        for m, v in zip(need, vecs[1:]):
            m["vec"] = [round(x, 5) for x in v]
        ranked = sorted(old, key=lambda m: -cosine(vecs[0], m["vec"]))
        return [m["text"] for m in ranked[:self.cfg["recall"]]]

    def narrate(self, action, recalled):
        s, cfg = self.s, self.cfg
        words = LENGTHS[cfg["length"]]
        system = ("You are the narrator of an interactive story. Genre: %s. Tone: %s.\nWorld: %s\n"
                  "The player is %s. Write in second person, present tense, about %d words in one or two "
                  "short paragraphs. Describe only what happens as a result of the player's action and end "
                  "at a moment that calls for a decision. Never list options, never use markdown, never "
                  "speak outside the story." % (s["preset"], s["tone"], s["setting"], s["role"], words))
        parts = []
        if s["summary"]:
            parts.append("Story so far: " + s["summary"])
        if recalled:
            parts.append("Earlier events that may matter:\n" + "\n".join("- " + brief(r, 200) for r in recalled))
        if s["turns"]:
            parts.append("Previous scene: " + brief(s["turns"][-1]["scene"], 500))
        parts.append("Current state - " + self.state_line())
        if s["characters"]:
            parts.append("Known characters: " + ", ".join(s["characters"]))
        if self.enhanced():
            parts.append("Persistent world state (authoritative; only explicit consequences change it): "
                         + self.world_context())
            if s.get("events"):
                parts.append("Persistent events:\n" + "\n".join(
                    "- " + brief(e.get("event", ""), 180) for e in s["events"][-5:]))
            profiles = []
            for name in s.get("characters", []):
                notes = self.character_memories(name)
                relation = s.get("relationships", {}).get(name, {})
                line = name
                if notes:
                    line += " remembers: " + "; ".join(note["text"] for note in notes)
                if relation:
                    line += " (trust %d, affinity %d, respect %d/100)" % (
                        relation.get("trust", 0), relation.get("affinity", 0), relation.get("respect", 0))
                    if relation.get("notes"):
                        line += "; relationship: " + "; ".join(relation["notes"][-2:])
                if s.get("character_goals", {}).get(name):
                    line += "; current goal: " + s["character_goals"][name]
                if s.get("character_motivations", {}).get(name):
                    line += "; motivations: " + ", ".join(s["character_motivations"][name][:3])
                if s.get("character_priorities", {}).get(name):
                    line += "; priorities: " + " > ".join(s["character_priorities"][name][:5])
                if s.get("character_emotions", {}).get(name):
                    line += "; emotional state: " + json.dumps(s["character_emotions"][name])
                if s.get("character_arcs", {}).get(name):
                    line += "; arc: " + json.dumps(s["character_arcs"][name])
                knowledge = s.get("character_knowledge", {}).get(name, [])
                if knowledge:
                    line += "; private knowledge (not shared by other characters): " + json.dumps(knowledge[-4:])
                profiles.append("- " + line)
            if profiles:
                parts.append("Persistent character profiles:\n" + "\n".join(profiles))
            if s.get("player_knowledge"):
                parts.append("What the player knows (do not confuse with NPC knowledge): " +
                             json.dumps(s["player_knowledge"][-8:], ensure_ascii=True))
            parts.append("Character knowledge is private: a character may use or reveal only facts listed "
                         "under that character. Never leak another character's private facts. Keep world "
                         "facts, unresolved threads, scheduled events, character arcs, and relationships consistent. "
                         "Characters have independent goals and priorities; they may initiate plausible actions "
                         "or scheduled events when these goals call for it. Reintroduce an active thread when it "
                         "fits the current scene, without forcing every thread into every scene.")
        if action is None:
            parts.append("Begin the story. Introduce the setting and the player's situation at %s, "
                         "with the goal: %s." % (s["location"], s["objective"]))
        else:
            parts.append("The player's action: %s\nContinue the story." % action)
            if self.enhanced() and action.lower().startswith("the player asks to trade:"):
                parts.append("Let the relevant character answer the trade proposal naturally. Only narrate an "
                             "exchange if that character agrees; make refused or unresolved offers clear.")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]
        wrap = Wrap(term_width())
        res = self.ai.chat("narrator", msgs, "narrate", stream=wrap.feed if cfg["stream"] else None,
                           options={"temperature": cfg["temperature"], "num_predict": int(words * 1.9) + 40})
        if not cfg["stream"]:
            wrap.feed(res["content"])
        wrap.close()
        text = tidy(res["content"])
        if not text:
            raise OllamaError("the narrator returned nothing", 500)
        return text

    def analyze(self, scene):
        prompt = ("Read this story passage and fill in: mood; danger (0 safe, 1 uneasy, 2 risky, 3 deadly); "
                  "characters (people or creatures present, at most 4, short names); location (at most 5 "
                  "words).\n\nPassage:\n" + scene)
        enhanced = self.enhanced()
        schema = ANALYSIS_SCHEMA
        if enhanced:
            prompt += ("\n\nAlso provide one concise persistent event summary. For each character in the scene, "
                       "record only supported changes to goals, motivations, knowledge, emotions, and arcs. "
                       "Knowledge is private: include only facts that character witnessed or was told, with "
                       "confidence 0-100 and source. Do not give one character another's knowledge. For new "
                       "memories, estimate importance, emotional strength, confidence, valence toward the player, "
                       "and whether this character wants to remember it. Update trust, affinity, and respect by "
                       "small deltas (-10..10); emotion changes are also deltas. Propose world changes only when "
                       "the passage explicitly causes them. Add or resolve plot threads and foreshadowing only "
                       "when supported. For world_changes use only: world fields weather/political_state/date/"
                       "alert_level; location fields condition/owner/details/known_event/guards; faction fields "
                       "influence or relation:OtherFaction; reputation fields reputation/fear/respect/fame/"
                       "notoriety/trust with a delta; object fields owner/condition; fact_add/fact_resolve for "
                       "verified world truths; thread_add with importance "
                       "low/medium/high/hidden or thread_resolve; foreshadow_add/reference/resolve/abandon; "
                       "schedule with an NPC target, action value and positive due_minutes. Include a reason. "
                       "Track player knowledge separately from NPC knowledge. Schedule future actions only when a "
                       "clear cause and delay exist. Advance story time plausibly. Current persistent state: %s" %
                       json.dumps({"world": self.s.get("world_state", {}),
                                   "relationships": self.s.get("relationships", {}),
                                   "character_goals": self.s.get("character_goals", {}),
                                   "character_priorities": self.s.get("character_priorities", {}),
                                   "character_emotions": self.s.get("character_emotions", {}),
                                   "character_arcs": self.s.get("character_arcs", {})}, ensure_ascii=True)[:3000])
            schema = ENHANCED_ANALYSIS_SCHEMA
        res = self.ai.chat("analyst", [{"role": "user", "content": prompt}], "analyse",
                           fmt=schema, options={"temperature": 0.1, "num_predict": 700 if enhanced else 120})
        d = json.loads(res["content"])
        mood = d.get("mood") if d.get("mood") in MOODS else "calm"
        try:
            danger = max(0, min(3, int(d.get("danger", 0))))
        except (TypeError, ValueError):
            danger = 0
        chars = [sstr(c, 30) for c in d.get("characters", []) if sstr(c, 30)][:4]
        loc = sstr(d.get("location", ""), 40)
        result = {"mood": mood, "danger": danger, "characters": chars,
                  "location": loc if 0 < len(loc.split()) <= 5 else ""}
        if enhanced:
            result["event"] = sstr(d.get("event"), 400) or brief(scene, 300)
            result["character_notes"] = [
                {"name": sstr(note.get("name"), 30), "memory": sstr(note.get("memory"), 180),
                 "wants_remember": bool(note.get("wants_remember", True)),
                 "importance": bounded_int(note.get("importance"), 50, 0, 100),
                 "emotional_strength": bounded_int(note.get("emotional_strength"), 40, 0, 100),
                 "confidence": bounded_int(note.get("confidence"), 80, 0, 100),
                 "source": sstr(note.get("source", "observed"), 60),
                 "valence": bounded_int(note.get("valence"), 0, -100, 100)}
                for note in d.get("character_notes", []) if isinstance(note, dict)
                and sstr(note.get("name"), 30) and sstr(note.get("memory"), 180)][:4]
            result["relationships"] = [
                {"character": sstr(rel.get("character"), 30),
                 "trust_change": rel.get("trust_change", 0),
                 "affinity_change": rel.get("affinity_change", 0),
                 "respect_change": rel.get("respect_change", 0),
                 "note": sstr(rel.get("note"), 120)}
                for rel in d.get("relationships", []) if isinstance(rel, dict)
                and sstr(rel.get("character"), 30)][:4]
            result["character_updates"] = []
            for update in d.get("character_updates", []) if isinstance(d.get("character_updates"), list) else []:
                if not isinstance(update, dict) or not sstr(update.get("name"), 30):
                    continue
                knowledge = update.get("knowledge", [])
                result["character_updates"].append({
                    "name": sstr(update.get("name"), 30), "goal": sstr(update.get("goal"), 160),
                    "motivations": [sstr(x, 100) for x in update.get("motivations", [])
                                    if sstr(x, 100)][:4] if isinstance(update.get("motivations"), list) else [],
                    "priorities": [sstr(x, 100) for x in update.get("priorities", [])
                                   if sstr(x, 100)][:5] if isinstance(update.get("priorities"), list) else [],
                    "knowledge": [{"fact": sstr(item.get("fact"), 180),
                                   "confidence": bounded_int(item.get("confidence"), 50, 0, 100),
                                   "source": sstr(item.get("source"), 60)}
                                  for item in knowledge if isinstance(item, dict) and sstr(item.get("fact"), 180)]
                                 if isinstance(knowledge, list) else [],
                    "emotion_changes": {key: bounded_int(update.get(key + "_change"), 0, -20, 20)
                                        for key in ("anger", "fear", "stress", "confidence")},
                    "arc": sstr(update.get("arc"), 100), "arc_stage": sstr(update.get("arc_stage"), 100),
                })
            changes = d.get("world_changes", [])
            result["world_changes"] = [{
                "kind": sstr(change.get("kind"), 30), "target": sstr(change.get("target"), 100),
                "field": sstr(change.get("field"), 40), "value": sstr(change.get("value"), 180),
                "delta": bounded_int(change.get("delta"), 0, -100, 100),
                "reason": sstr(change.get("reason"), 160),
                "due_minutes": bounded_int(change.get("due_minutes"), 0, 0, 10080),
            } for change in changes if isinstance(change, dict)][:12] if isinstance(changes, list) else []
            result["time_advance_minutes"] = bounded_int(d.get("time_advance_minutes"), 10, 0, 1440)
            result["scene_objective"] = sstr(d.get("scene_objective"), 160)
            result["scene_stakes"] = sstr(d.get("scene_stakes"), 120)
            facts = d.get("player_knowledge", [])
            result["player_knowledge"] = [{"fact": sstr(item.get("fact"), 180),
                                            "confidence": bounded_int(item.get("confidence"), 80, 0, 100),
                                            "source": sstr(item.get("source"), 60)}
                                           for item in facts if isinstance(item, dict)
                                           and sstr(item.get("fact"), 180)][:8] if isinstance(facts, list) else []
        return result

    def summarize(self, new_turn):
        s = self.s
        pending = (s["turns"] + [new_turn])[s["summarized_upto"]:]
        if len(pending) <= 3:
            return None
        fold = pending[:-2]
        text = "\n".join("Player: %s\n%s" % (t["action"], brief(t["scene"], 350)) for t in fold)[-1800:]
        if self.enhanced():
            active = [t["title"] for t in self.ensure_world_state()["threads"] if t.get("status") == "active"]
            if active:
                text += "\nUnresolved threads to retain: " + "; ".join(active[-8:])
        prompt = ("Current summary:\n%s\n\nNew events:\n%s\n\nWrite an updated summary of the whole story so "
                  "far in at most 70 words. Keep names, places, items and unresolved threads. Plain text "
                  "only." % (s["summary"] or "(none yet)", text))
        res = self.ai.chat("analyst", [{"role": "user", "content": prompt}], "summary",
                           options={"temperature": 0.2, "num_predict": 150})
        summary = tidy(res["content"])
        return (summary, s["summarized_upto"] + len(fold)) if summary else None

    def make_choices(self, scene):
        n = self.cfg["choices"]
        system = "You write the next-action menu for an interactive story. Reply with JSON only."
        user = ("Scene:\n%s\n\nState: %s\n\nList %d different short actions (at most 8 words each, starting "
                "with a verb) the player could take next. Make them distinct: vary cautious, bold and "
                "clever." % (scene, self.state_line(), n))
        if self.enhanced():
            world = self.ensure_world_state()
            active = [thread["title"] for thread in world["threads"] if thread.get("status") == "active"]
            user += "\nScene objective and stakes: %s / %s. Active threads: %s. Honor current world facts." % (
            world["scene"].get("objective", ""), world["scene"].get("stakes", ""),
            "; ".join(active[-5:]) or "none")
        schema = {"type": "object", "required": ["choices"],
                  "properties": {"choices": {"type": "array", "items": {"type": "string"},
                                             "minItems": n, "maxItems": n}}}
        res = self.ai.chat("choices", [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           "choices", fmt=schema, options={"temperature": 0.7, "num_predict": 30 * n + 20})
        d = json.loads(res["content"])
        return clean_choices(d.get("choices") if isinstance(d, dict) else None, n)

    def track(self, action, scene):
        system = ("You are a game state tracker. Read the passage and call a function for every change it "
                  "describes (items gained or lost, a new location, health changes, a new goal). If nothing "
                  "changed, call no function.")
        tools = TOOLS
        if self.enhanced():
            system += (" A trade is complete only if the passage explicitly says the character agreed and "
                       "the exchange happened. Then call trade_items with both items and the character. "
                       "Never call it for a proposal or a refused trade.")
        else:
            tools = TOOLS[:-1]
        user = "Player action: %s\nPassage: %s\nCurrent state: %s" % (action or "(story begins)", scene, self.state_line())
        res = self.ai.chat("tools", [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           "track", tools=tools, options={"temperature": 0, "num_predict": 160})
        return res["tool_calls"]

    def apply_calls(self, calls):
        s, notes = self.s, []
        for call in calls[:6]:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            name, args = fn.get("name"), fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                continue
            if name == "add_item":
                it = sstr(args.get("item"), 30).lower()
                if it and it not in s["items"] and len(s["items"]) < 12:
                    s["items"].append(it)
                    notes.append("+" + it)
            elif name == "remove_item":
                it = sstr(args.get("item"), 30).lower()
                for have in list(s["items"]):
                    if it and (it in have or have in it):
                        s["items"].remove(have)
                        notes.append("-" + have)
                        break
            elif name == "change_location":
                place = sstr(args.get("place"), 40)
                if place and place.lower() != s["location"].lower():
                    s["location"] = place
                    if place not in s.setdefault("places", []):
                        s["places"].append(place)
                    notes.append("at " + place)
            elif name == "adjust_health":
                try:
                    amt = max(-5, min(5, int(float(args.get("amount")))))
                except (TypeError, ValueError):
                    continue
                if amt:
                    s["hp"] = max(0, min(s["max_hp"], s["hp"] + amt))
                    notes.append("hp %+d" % amt)
            elif name == "set_objective":
                goal = sstr(args.get("objective"), 80)
                if goal and goal.lower() != s["objective"].lower():
                    s["objective"] = goal
                    notes.append("new goal")
            elif name == "trade_items" and self.enhanced():
                character = sstr(args.get("character"), 30)
                give = sstr(args.get("give_item"), 30).lower()
                receive = sstr(args.get("receive_item"), 30).lower()
                have = next((item for item in s["items"] if item.lower() == give), None)
                if character and have and receive and receive != give and len(s["items"]) < 12 \
                        and receive not in [item.lower() for item in s["items"]]:
                    s["items"].remove(have)
                    s["items"].append(receive)
                    key = self.character_key(character)
                    s.setdefault("character_items", {}).setdefault(key, []).append(have)
                    notes.append("trade with %s: -%s +%s" % (key, have, receive))
        return notes

    def ensure_world_state(self):
        world = self.s.setdefault("world_state", default_world_state(self.s["location"]))
        defaults = default_world_state(self.s["location"])
        for key, value in defaults.items():
            world.setdefault(key, value)
        world.setdefault("locations", {}).setdefault(self.s["location"], {
            "condition": "intact", "owner": "", "details": "", "known_events": [], "guards": 0})
        return world

    def record_world_event(self, description, cause_id=""):
        world = self.ensure_world_state()
        event = {"turn": len(self.s["turns"]) + 1, "description": sstr(description, 240)}
        if event["description"]:
            world["world_events"].append(event)
            del world["world_events"][:-100]
            if cause_id:
                world["causality"].append({"cause": cause_id, "effect": event["description"],
                                           "turn": event["turn"]})
                del world["causality"][:-200]

    def apply_world_change(self, change):
        world = self.ensure_world_state()
        kind, target = change["kind"], change["target"] or change["value"]
        field, value = change["field"], change["value"]
        delta, reason = change["delta"], change["reason"]
        description = ""
        if kind == "world":
            if field in ("weather", "political_state", "date") and value:
                world[field] = value
                description = "%s changed to %s" % (field.replace("_", " "), value)
            elif field == "alert_level":
                world[field] = bounded_int(world.get(field, 0) + (delta or bounded_int(value)), 0, 0, 5)
                description = "city alert level is now %d" % world[field]
        elif kind == "location" and target:
            place = world["locations"].setdefault(target, {
                "condition": "intact", "owner": "", "details": "", "known_events": [], "guards": 0})
            if field in ("condition", "owner", "details") and value:
                place[field] = value
                description = "%s: %s is now %s" % (target, field, value)
            elif field == "guards":
                place[field] = bounded_int(value, bounded_int(place.get(field), 0, 0, 10000), 0, 10000)
                description = "%s has %d guards" % (target, place[field])
            elif field == "known_event" and value:
                place.setdefault("known_events", []).append(value)
                del place["known_events"][:-12]
                description = "%s: %s" % (target, value)
            if target not in self.s.setdefault("places", []):
                self.s["places"].append(target)
        elif kind == "faction" and target:
            faction = world["factions"].setdefault(target, {"influence": 0, "relations": {}})
            if field == "influence":
                faction["influence"] = bounded_int(faction.get("influence", 0) + delta, 0, 0, 100)
                description = "%s influence is now %d" % (target, faction["influence"])
            elif field.startswith("relation:"):
                other = field.split(":", 1)[1][:60]
                if other:
                    faction["relations"][other] = bounded_int(
                        faction["relations"].get(other, 0) + delta, 0, -100, 100)
                    description = "%s relations with %s changed" % (target, other)
        elif kind == "reputation" and target:
            stat = field if field in ("reputation", "fear", "respect", "fame", "notoriety", "trust") else "reputation"
            standing = world["reputation"].setdefault(target, {})
            standing[stat] = bounded_int(standing.get(stat, 0) + delta, 0, -100, 100)
            description = "%s %s is now %+d" % (target, stat, standing[stat])
        elif kind == "object" and target:
            obj = world["objects"].setdefault(target, {"id": uuid.uuid4().hex[:10], "owner": "unknown",
                                                       "condition": "intact", "history": []})
            if field in ("owner", "condition") and value:
                obj[field] = value
                obj.setdefault("history", []).append({"turn": len(self.s["turns"]) + 1,
                                                       "event": reason or "%s became %s" % (field, value)})
                del obj["history"][:-12]
                description = "%s is now owned by %s" % (target, value) if field == "owner" else \
                    "%s condition is now %s" % (target, value)
        elif kind == "fact_add" and target:
            if target.lower() not in [fact["text"].lower() for fact in world["facts"]]:
                world["facts"].append({"text": target[:180], "confidence": bounded_int(field, 100, 0, 100),
                                       "source": reason[:120], "turn": len(self.s["turns"]) + 1,
                                       "status": "active"})
                description = "world fact established: %s" % target
        elif kind == "fact_resolve" and target:
            fact = next((fact for fact in world["facts"] if fact["text"].lower() == target.lower()), None)
            if fact and fact["status"] == "active":
                fact["status"] = "resolved"
                description = "world fact resolved: %s" % target
        elif kind in ("thread_add", "thread_resolve") and target:
            threads = world["threads"]
            thread = next((t for t in threads if t["title"].lower() == target.lower()), None)
            if kind == "thread_add" and thread is None:
                importance = field.lower() if field.lower() in ("low", "medium", "high", "hidden") else "medium"
                threads.append({"id": uuid.uuid4().hex[:8], "title": target[:120], "importance": importance,
                                "participants": self.s.get("characters", [])[-4:],
                                "started": len(self.s["turns"]) + 1,
                                "last_progress": len(self.s["turns"]) + 1, "status": "active",
                                "possible_resolution": reason[:160]})
                description = "plot thread opened: %s" % target
            elif kind == "thread_resolve" and thread and thread["status"] == "active":
                thread["status"] = "resolved"
                thread["last_progress"] = len(self.s["turns"]) + 1
                description = "plot thread resolved: %s" % target
        elif kind in ("foreshadow_add", "foreshadow_reference", "foreshadow_resolve", "foreshadow_abandon") and target:
            foreshadowing = world["foreshadowing"]
            hint = next((f for f in foreshadowing if f["text"].lower() == target.lower()), None)
            if kind == "foreshadow_add" and hint is None:
                foreshadowing.append({"text": target[:160], "introduced": len(self.s["turns"]) + 1,
                                      "status": "active", "references": 0})
                description = "foreshadowing introduced: %s" % target
            elif kind == "foreshadow_reference" and hint and hint["status"] == "active":
                hint["references"] = bounded_int(hint.get("references", 0) + 1, 0, 0, 1000)
                description = "foreshadowing revisited: %s" % target
            elif kind in ("foreshadow_resolve", "foreshadow_abandon") and hint and hint["status"] == "active":
                hint["status"] = "resolved" if kind == "foreshadow_resolve" else "abandoned"
                description = "foreshadowing %s: %s" % (hint["status"], target)
        elif kind == "schedule" and target and value and change["due_minutes"] > 0:
            world["pending_events"].append({"character": target[:60], "description": value[:180],
                "due_at": world["time_minutes"] + change["due_minutes"], "cause": reason[:160]})
            description = "scheduled: %s" % value
        if description:
            cause_id = self.s.get("events", [{}])[-1].get("id", "")
            self.record_world_event(description, cause_id)

    def advance_story_time(self, minutes):
        world = self.ensure_world_state()
        old_day = world["time_minutes"] // 1440
        world["time_minutes"] += bounded_int(minutes, 10, 0, 720)
        new_day = world["time_minutes"] // 1440
        if new_day != old_day:
            date_text = str(world.get("date", ""))
            parsed = None
            for date_format in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
                try:
                    parsed = datetime.datetime.strptime(date_text, date_format)
                    break
                except ValueError:
                    continue
            if parsed:
                world["date"] = (parsed + datetime.timedelta(days=new_day - old_day)).strftime(date_format)
            elif date_text.startswith("Day "):
                day = bounded_int(date_text[4:], 1, 1, 1000000)
                world["date"] = "Day %d" % (day + new_day - old_day)
        due, upcoming = [], []
        for event in world["pending_events"]:
            if event.get("due_at", 0) <= world["time_minutes"]:
                due.append(event)
            else:
                upcoming.append(event)
        world["pending_events"] = upcoming[-30:]
        for event in due:
            message = "Off-screen: %s %s" % (event.get("character", "Someone"), event.get("description", "acts"))
            self.record_world_event(message, self.s.get("events", [{}])[-1].get("id", ""))

    def apply_character_updates(self, updates):
        s, turn = self.s, len(self.s["turns"]) + 1
        for update in updates:
            name = self.character_key(update["name"])
            if name.lower() not in [c.lower() for c in s["characters"]] and len(s["characters"]) < 8:
                s["characters"].append(name)
            if update["goal"]:
                s["character_goals"][name] = update["goal"]
            if update["motivations"]:
                s["character_motivations"][name] = update["motivations"]
            if update.get("priorities"):
                s.setdefault("character_priorities", {})[name] = update["priorities"]
            knowledge = s["character_knowledge"].setdefault(name, [])
            for fact in update["knowledge"]:
                if fact["fact"].lower() not in [k["fact"].lower() for k in knowledge]:
                    knowledge.append(dict(fact, turn=turn))
            del knowledge[:-20]
            emotions = s["character_emotions"].setdefault(name, {
                "anger": 0, "fear": 0, "stress": 0, "confidence": 50})
            for emotion, change in update["emotion_changes"].items():
                emotions[emotion] = bounded_int(emotions.get(emotion, 50 if emotion == "confidence" else 0)
                                                + change, 0, 0, 100)
            if update["arc"]:
                arc = s["character_arcs"].setdefault(name, {"name": update["arc"], "stage": "", "history": []})
                if arc.get("name") != update["arc"]:
                    arc["history"].append({"turn": turn, "stage": arc.get("stage", ""), "event": "arc changed"})
                    arc["name"] = update["arc"]
                if update["arc_stage"] and update["arc_stage"] != arc.get("stage"):
                    arc["history"].append({"turn": turn, "stage": update["arc_stage"],
                                            "event": "stage advanced"})
                    arc["stage"] = update["arc_stage"]
                del arc["history"][:-12]

    def apply_analysis(self, info):
        s = self.s
        self.ensure_world_state()
        s.setdefault("character_notes", {})
        s.setdefault("relationships", {})
        s.setdefault("relationship_history", [])
        s.setdefault("character_goals", {})
        s.setdefault("character_motivations", {})
        s.setdefault("character_priorities", {})
        s.setdefault("character_knowledge", {})
        s.setdefault("player_knowledge", [])
        s.setdefault("character_emotions", {})
        s.setdefault("character_arcs", {})
        for relationship in info.get("relationships", []):
            name = self.character_key(relationship["character"])
            relation = s["relationships"].setdefault(name, {
                "trust": 0, "affinity": 0, "respect": 0, "notes": []})
            changes = {}
            for dimension in ("trust", "affinity", "respect"):
                delta = bounded_int(relationship.get(dimension + "_change"), 0, -10, 10)
                relation[dimension] = bounded_int(relation.get(dimension, 0) + delta, 0, -100, 100)
                changes[dimension] = delta
            note = relationship.get("note", "")
            if note:
                relation.setdefault("notes", []).append(note)
                del relation["notes"][:-8]
            if note or any(changes.values()):
                s["relationship_history"].append({"turn": len(s["turns"]) + 1, "character": name,
                    "changes": changes, "note": note, "scores": {k: relation[k] for k in changes}})
        del s["relationship_history"][:-100]
        self.apply_character_updates(info.get("character_updates", []))
        for fact in info.get("player_knowledge", []):
            known = s["player_knowledge"]
            if fact["fact"].lower() not in [entry["fact"].lower() for entry in known]:
                known.append(dict(fact, turn=len(s["turns"]) + 1))
        del s["player_knowledge"][:-100]
        current_turn = len(s["turns"]) + 1
        for name, memories in list(s["character_notes"].items()):
            key = self.character_key(name)
            relation = s["relationships"].get(key, {})
            normalized = [normalize_memory(item, 0) for item in memories]
            normalized = [m for m in normalized if m and not (
                current_turn - m["turn"] > 30
                and memory_retention_score(m, relation, current_turn) < 0.025)]
            normalized.sort(key=lambda m: memory_retention_score(m, relation, current_turn), reverse=True)
            s["character_notes"][key] = normalized[:12]
            if key != name:
                s["character_notes"].pop(name, None)
        for entry in info.get("character_notes", []):
            name = self.character_key(entry["name"])
            memories = s["character_notes"].setdefault(name, [])
            normalized = [normalize_memory(item, 0) for item in memories]
            record = normalize_memory(dict(entry, turn=current_turn))
            if record and not any(m["text"].lower() == record["text"].lower() for m in normalized if m):
                normalized.append(record)
            relation = s["relationships"].get(name, {})
            normalized = [m for m in normalized if m and not (
                current_turn - m["turn"] > 30
                and memory_retention_score(m, relation, current_turn) < 0.025)]
            normalized.sort(key=lambda m: memory_retention_score(m, relation, current_turn), reverse=True)
            s["character_notes"][name] = normalized[:12]
        for change in info.get("world_changes", []):
            self.apply_world_change(change)
        world = self.ensure_world_state()
        world["scene"] = {"objective": info.get("scene_objective", ""),
                           "stakes": info.get("scene_stakes", ""),
                           "participants": info.get("characters", [])}
        self.advance_story_time(info.get("time_advance_minutes", 10))

    def epilogue(self):
        s = self.s
        msgs = [{"role": "system", "content": "You are the narrator of an interactive %s story." % s["preset"]},
                {"role": "user", "content": "Story so far: %s\nLast scene: %s\nThe player's health has reached "
                 "zero. Write a short final paragraph (about 40 words) that ends the story. Do not offer "
                 "choices." % (s["summary"] or "(none)", brief(s["turns"][-1]["scene"], 400))}]
        wrap = Wrap(term_width())
        res = self.ai.chat("narrator", msgs, "epilogue", stream=wrap.feed if self.cfg["stream"] else None,
                           options={"temperature": self.cfg["temperature"], "num_predict": 100})
        if not self.cfg["stream"]:
            wrap.feed(res["content"])
        wrap.close()
        return tidy(res["content"])

    # ---- one full turn
    def take_turn(self, action):
        s, ai, cfg = self.s, self.ai, self.cfg
        ai.timing = {}
        t0 = time.time()
        recalled = []
        if action is not None and cfg["recall"] and ai.has("memory") and len(s["memories"]) > 2:
            recalled = self.guard(self.recall, action, role="memory") or []
        say()
        scene = self.narrate(action, recalled)
        info = self.guard(self.analyze, scene, role="analyst") or {}
        turn = {"action": action or "(the story begins)", "scene": scene,
                "mood": info.get("mood", "calm"), "danger": info.get("danger", 0)}
        folded = self.guard(self.summarize, turn, role="analyst")
        choices = self.guard(self.make_choices, scene, role="choices") or []
        calls = []
        if cfg["tracking"] and ai.has("tools"):
            calls = self.guard(self.track, action, scene, role="tools") or []
        # ---- commit (nothing above touched the saved state, so Ctrl-C leaves it intact)
        notes = self.apply_calls(calls)
        if self.enhanced():
            event = {
                "id": uuid.uuid4().hex[:12],
                "turn": len(s["turns"]) + 1,
                "action": action or "(the story begins)",
                "event": (info.get("event") if info else "") or brief(scene, 300),
                "characters": info.get("characters", []) if info else [],
            }
            s.setdefault("events", []).append(event)
            del s["events"][:-100]
        if info:
            for c in info["characters"]:
                if c.lower() not in [x.lower() for x in s["characters"]]:
                    s["characters"].append(c)
            s["characters"] = s["characters"][-8:]
            if self.enhanced():
                self.apply_analysis(info)
                s["characters"] = s["characters"][-8:]
            if info["location"] and not (cfg["tracking"] and ai.has("tools")):
                s["location"] = info["location"]
                if info["location"] not in s.setdefault("places", []):
                    s["places"].append(info["location"])
        if folded:
            s["summary"], s["summarized_upto"] = folded
        s["turns"].append(turn)
        s["memories"].append({"text": brief("%s -> %s" % (turn["action"], scene), 600), "vec": None})
        extra = FALLBACK_CHOICES[:]
        while len(choices) < max(2, cfg["choices"]) and extra:
            c = extra.pop(0)
            if c.lower() not in [x.lower() for x in choices]:
                choices.append(c)
        s["choices"] = choices
        if info:
            say(dim("  [%s, danger %d/3]" % (info["mood"], info["danger"])))
        if notes:
            say(yellow("  (%s)" % ", ".join(notes)))
        if s["hp"] <= 0:
            say()
            ep = self.guard(self.epilogue)
            if ep:
                s["turns"].append({"action": "(the end)", "scene": ep, "mood": "sad", "danger": 0})
            s["over"] = True
            say()
            say(bold("  THE END"))
        self.app.save_state(s)
        if cfg["timing"]:
            parts = " | ".join("%s %.1f" % (k, v) for k, v in ai.timing.items())
            say(dim("  turn %.1fs: %s" % (time.time() - t0, parts)))

    # ---- player commands
    def route(self, text):
        t = text.lower().strip(" .!?")
        for intent, words in KEYWORDS.items():
            if t in words:
                return intent
        if re.search(r"\b(trade|barter|exchange|swap)\b", t):
            return "trade"
        if t.startswith(("talk to ", "speak to ", "chat with ")):
            return "talk"
        if '"' in text or t.startswith(("say ", "ask ", "tell ")):
            return "talk"
        if len(t.split()) < 2 or not self.ai.has("router"):
            return "action"
        return self.guard(self.classify, text, role="router") or "action"

    def classify(self, text):
        res = self.ai.chat("router", [{"role": "user", "content": ROUTE_PROMPT + text}], "route",
                           options={"temperature": 0, "num_predict": 4})
        return parse_intent(res["content"])

    def look(self):
        s = self.s
        say()
        if not self.ai.has("router"):
            say(dim("  " + brief(s["turns"][-1]["scene"], 400)))
            return
        msgs = [{"role": "system", "content": "You describe what the player currently sees in two short "
                 "sentences. Do not invent events, characters or items."},
                {"role": "user", "content": "Setting: %s\nLocation: %s\nLast scene: %s" % (
                    s["setting"], s["location"], brief(s["turns"][-1]["scene"], 500))}]
        wrap = Wrap(term_width())
        res = self.ai.chat("router", msgs, "look", stream=wrap.feed if self.cfg["stream"] else None,
                           options={"temperature": 0.5, "num_predict": 70})
        if not self.cfg["stream"]:
            wrap.feed(res["content"])
        wrap.close()

    def show_items(self):
        s = self.s
        say()
        say("  Role:      %s" % s["role"])
        say("  Location:  %s" % s["location"])
        say("  Health:    %d/%d" % (s["hp"], s["max_hp"]))
        say("  Carrying:  %s" % (", ".join(s["items"]) or "nothing"))
        say("  Goal:      %s" % s["objective"])
        if s["characters"]:
            say("  Met:       %s" % ", ".join(s["characters"]))
        if self.enhanced():
            world = self.ensure_world_state()
            say("  World:     %s | %s | alert %d/5" % (
                self.clock_label(), world.get("weather", "unknown"), world.get("alert_level", 0)))

    def show_debug(self):
        world = self.ensure_world_state()
        say()
        say(rule("Story state / debug"))
        say("  Timeline: %s (parent: %s)" % (
            world.get("timeline", {}).get("name", "A"),
            world.get("timeline", {}).get("parent") or "none"))
        say("  Time: %s | weather: %s | alert: %s/5 | politics: %s" % (
            self.clock_label(), world.get("weather", "unknown"), world.get("alert_level", 0),
            world.get("political_state", "unknown")))
        say("  Location: %s | state: %s" % (
            self.s["location"], json.dumps(world["locations"].get(self.s["location"], {}), ensure_ascii=True)))
        for name in self.s.get("characters", []):
            relation = self.s.get("relationships", {}).get(name, {})
            emotion = self.s.get("character_emotions", {}).get(name, {})
            memories = self.character_memories(name, 3)
            say("  %s: trust %+d affinity %+d respect %+d | goal: %s | emotion: %s" % (
                name, relation.get("trust", 0), relation.get("affinity", 0), relation.get("respect", 0),
                self.s.get("character_goals", {}).get(name, "none"),
                json.dumps(emotion, ensure_ascii=True) if emotion else "unknown"))
            if memories:
                say("    retained memories: %s" % "; ".join(m["text"] for m in memories))
            motivations = self.s.get("character_motivations", {}).get(name, [])
            priorities = self.s.get("character_priorities", {}).get(name, [])
            if motivations or priorities:
                say("    motivations: %s | priorities: %s" % (
                    ", ".join(motivations) or "none", " > ".join(priorities) or "none"))
            knowledge = self.s.get("character_knowledge", {}).get(name, [])
            if knowledge:
                say("    private knowledge: %s" % json.dumps(knowledge[-4:], ensure_ascii=True))
        active_threads = [t for t in world["threads"] if t.get("status") == "active"]
        say("  Active threads (%d): %s" % (len(active_threads),
            "; ".join("[%s] %s" % (t["importance"], t["title"]) for t in active_threads) or "none"))
        say("  Foreshadowing: %s" % ("; ".join(f["text"] for f in world["foreshadowing"]
              if f.get("status") == "active") or "none"))
        say("  Scheduled: %s" % ("; ".join(e["description"] for e in world["pending_events"]) or "none"))
        say("  Recent causes: %s" % ("; ".join(c["effect"] for c in world["causality"][-5:]) or "none"))

    def branch_timeline(self):
        parent = self.s
        parent_timeline = self.ensure_world_state().setdefault("timeline", {"name": "A", "branch_count": 0})
        branch_number = bounded_int(parent_timeline.get("branch_count"), 0, 0, 999) + 1
        parent_timeline["branch_count"] = branch_number
        self.app.save_state(parent)
        branch = copy.deepcopy(parent)
        branch["id"] = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        branch["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        branch["updated"] = branch["created"]
        parent_name = parent_timeline.get("name", "A")
        branch["world_state"]["timeline"] = {
            "name": "%s.%d" % (parent_name, branch_number), "parent": parent_name,
            "parent_save_id": parent["id"], "branch_count": 0}
        self.s = branch
        self.app.save_state(branch)
        say(green("  Branched timeline %s; the original save remains available." %
                  branch["world_state"]["timeline"]["name"]))

    def customize(self):
        while True:
            say()
            say(rule("Customize story"))
            say("  1) Places and setting")
            say("  2) Characters and memories")
            say("  3) Relationships")
            say(dim("  u/x back"))
            key = getkey()
            if key == "1":
                self.edit_places()
            elif key == "2":
                self.edit_characters()
            elif key == "3":
                self.edit_relationships()
            elif key in ("u", "x", "q", "esc", "\n"):
                self.app.save_state(self.s)
                return

    def edit_places(self):
        s = self.s
        places = s.setdefault("places", [])
        if s["location"] not in places:
            places.append(s["location"])
        while True:
            say()
            say(rule("Places"))
            say("  Setting: %s" % s["setting"])
            for i, place in enumerate(places, 1):
                say("  %d) %s%s" % (i, place, " (current)" if place == s["location"] else ""))
            say(dim("  number rename  t travel  a add  d delete  s edit setting  x back"))
            key = getkey()
            if key == "a":
                place = ask("  New place> ")[:80]
                if place and place.lower() not in [p.lower() for p in places]:
                    places.append(place)
            elif key == "s":
                setting = ask("  New world setting (Enter keeps current)> ")[:500]
                if setting:
                    s["setting"] = setting
            elif key in ("t", "d"):
                raw = ask("  Place number> ")
                if raw.isdigit() and 1 <= int(raw) <= len(places):
                    place = places[int(raw) - 1]
                    if key == "t":
                        s["location"] = place
                    elif len(places) > 1:
                        places.remove(place)
                        if s["location"] == place:
                            s["location"] = places[0]
            elif key.isdigit() and 1 <= int(key) <= len(places):
                old = places[int(key) - 1]
                place = ask("  Rename '%s' (Enter keeps)> " % old)[:80]
                if place:
                    places[int(key) - 1] = place
                    if s["location"] == old:
                        s["location"] = place
            elif key in ("x", "q", "esc", "\n"):
                return

    def edit_characters(self):
        s = self.s
        s.setdefault("character_notes", {})
        s.setdefault("relationships", {})
        s.setdefault("character_items", {})
        while True:
            say()
            say(rule("Characters"))
            for i, name in enumerate(s["characters"], 1):
                relation = s["relationships"].get(name, {})
                memories = self.character_memories(name, 2)
                say("  %d) %s%s" % (i, name, " - " + "; ".join(m["text"] for m in memories)
                                    if memories else ""))
                if relation:
                    say(dim("     trust %d  affinity %d  respect %d" % (
                        relation.get("trust", 0), relation.get("affinity", 0), relation.get("respect", 0))))
            say(dim("  number edit  a add  d delete  x back"))
            key = getkey()
            if key == "a":
                name = ask("  Character name> ")[:30]
                if name and name.lower() not in [c.lower() for c in s["characters"]]:
                    s["characters"].append(name)
                    s["character_notes"][name] = []
                    s["relationships"][name] = {"trust": 0, "affinity": 0, "respect": 0, "notes": []}
            elif key == "d":
                raw = ask("  Character number to delete> ")
                if raw.isdigit() and 1 <= int(raw) <= len(s["characters"]):
                    name = s["characters"].pop(int(raw) - 1)
                    for field in ("character_notes", "relationships", "character_items", "character_goals",
                                  "character_motivations", "character_priorities", "character_knowledge",
                                  "character_emotions", "character_arcs"):
                        s[field].pop(name, None)
            elif key.isdigit() and 1 <= int(key) <= len(s["characters"]):
                old = s["characters"][int(key) - 1]
                name = ask("  Rename '%s' (Enter keeps)> " % old)[:30] or old
                description = ask("  Add a memory or description (Enter keeps)> ")[:180]
                s["characters"][int(key) - 1] = name
                for field in ("character_notes", "relationships", "character_items", "character_goals",
                              "character_motivations", "character_priorities", "character_knowledge",
                              "character_emotions", "character_arcs"):
                    if old != name and old in s[field]:
                        s[field][name] = s[field].pop(old)
                if description:
                    s["character_notes"].setdefault(name, []).append(description)
                    del s["character_notes"][name][:-8]
                if name not in s["relationships"]:
                    s["relationships"][name] = {"trust": 0, "affinity": 0, "respect": 0, "notes": []}
            elif key in ("x", "q", "esc", "\n"):
                return

    def edit_relationships(self):
        s = self.s
        s.setdefault("relationships", {})
        while True:
            say()
            say(rule("Relationships"))
            for i, name in enumerate(s["characters"], 1):
                relation = s["relationships"].setdefault(name, {
                    "trust": 0, "affinity": 0, "respect": 0, "notes": []})
                say("  %d) %-20s trust %+d  affinity %+d  respect %+d" % (
                    i, name[:20], relation.get("trust", 0), relation.get("affinity", 0),
                    relation.get("respect", 0)))
            say(dim("  number edit scores/notes  x back"))
            key = getkey()
            if key.isdigit() and 1 <= int(key) <= len(s["characters"]):
                name = s["characters"][int(key) - 1]
                relation = s["relationships"].setdefault(name, {
                    "trust": 0, "affinity": 0, "respect": 0, "notes": []})
                for dimension in ("trust", "affinity", "respect"):
                    raw = ask("  %s [-100..100, current %+d]> " % (dimension, relation[dimension]))
                    if raw:
                        try:
                            relation[dimension] = max(-100, min(100, int(raw)))
                        except ValueError:
                            say(dim("  Enter a whole number; value unchanged."))
                note = ask("  Relationship note (Enter keeps)> ")[:120]
                if note:
                    relation.setdefault("notes", []).append(note)
                    del relation["notes"][:-8]
            elif key in ("x", "q", "esc", "\n"):
                return

    def recap(self):
        s = self.s
        say()
        w = Wrap(term_width())
        w.feed(s["summary"] or "Nothing to summarise yet. Last scene: " + brief(s["turns"][-1]["scene"], 300))
        w.close()

    def show_help(self):
        say()
        say("  1-%d  take that action        t  type any action (or 'look', 'items', ...)" % len(self.s["choices"]))
        say("  l  look around (fast)         i  items and status       r  story recap")
        say("  u  customize places/people    x  settings              b  benchmark models")
        say("  d  story state/debug          a  branch timeline       e  export as Markdown")
        say("  Talk to anyone or ask a character to trade")
        say("  q  save and return to menu    (every turn is saved automatically)")

    def menu(self):
        s = self.s
        say()
        say(rule("%s - turn %d" % (s["preset"], len(s["turns"]))))
        bar = "#" * s["hp"] + "." * (s["max_hp"] - s["hp"])
        say(" %s %s (%s)   %s [%s] %d/%d" % (bold("At:"), s["location"], self.clock_label(),
                              bold("HP"), bar, s["hp"], s["max_hp"]))
        say(" %s %s" % (bold("Carrying:"), ", ".join(s["items"]) or "nothing"))
        say(" %s %s" % (bold("Goal:"), s["objective"]))
        say()
        for i, c in enumerate(s["choices"], 1):
            say("  %d) %s" % (i, c))
        say(dim("  t:type  l:look  i:items  r:recap  u:edit  d:debug  a:branch"))
        say(dim("  x:settings  b:bench  e:export  h:help  q:menu"))

    def play(self):
        s = self.s
        if not s["turns"]:
            self.app.save_state(s)
            try:
                self.take_turn(None)
            except (OllamaError, Unavailable) as e:
                say(red("  Could not start: %s" % (e if isinstance(e, OllamaError) else "no narrator model")))
                return
            except KeyboardInterrupt:
                say(dim("\n  Cancelled."))
                return
        while True:
            if s["over"]:
                say()
                say(dim("  This story is finished. Press any key to return to the menu."))
                getkey()
                return
            self.menu()
            k = getkey()
            action = None
            if k.isdigit() and 1 <= int(k) <= len(s["choices"]):
                action = s["choices"][int(k) - 1]
            elif k == "t":
                text = ask("  Your action> ")
                if not text:
                    continue
                intent = self.route(text)
                if intent == "look":
                    self.look()
                elif intent == "inventory":
                    self.show_items()
                elif intent == "help":
                    self.show_help()
                elif intent == "quit":
                    return
                elif intent == "trade":
                    action = "The player asks to trade: %s" % text
                elif intent == "talk":
                    action = "The player talks to someone in the scene: %s" % text
                else:
                    action = text
            elif k == "l":
                self.look()
            elif k == "i":
                self.show_items()
            elif k == "r":
                self.recap()
            elif k == "h":
                self.show_help()
            elif k == "e":
                self.app.export(s)
            elif k == "x":
                self.app.settings_menu()
            elif k == "u":
                self.customize()
            elif k == "d":
                self.show_debug()
            elif k == "a":
                self.branch_timeline()
                s = self.s
            elif k == "b":
                self.app.benchmark_menu()
            elif k in ("q", "esc"):
                return
            if action is not None:
                try:
                    self.take_turn(action)
                except KeyboardInterrupt:
                    say(dim("\n  Turn cancelled - your choices are unchanged."))
                except (OllamaError, Unavailable) as e:
                    say(red("  Turn failed: %s" % (e if isinstance(e, OllamaError) else "model unavailable")))
                    say(dim("  Your choices are unchanged - try again, or check x > settings."))


# --------------------------------------------------------------------------- benchmark
SPEED_PROMPT = "Write one vivid sentence about a lighthouse in a storm."
SAMPLE_SCENE = ("You push open the cellar door. Cold air drifts up from the dark, carrying the smell of damp "
                "stone. Below, a faint light flickers, and something metallic scrapes across the floor.")


def _json_chat(c, m, o, ka, prompt, schema):
    r = c.chat(m, [{"role": "user", "content": prompt}], fmt=schema, options=o, keep_alive=ka)
    return json.loads(r["content"])


def t_narrator(c, m, o, ka):
    r = c.chat(m, [{"role": "user", "content": "Continue in one sentence: You open the cellar door."}],
               options=dict(o, num_predict=40), keep_alive=ka)
    return (1 if len(r["content"].split()) >= 4 else 0), 1


def t_choices(c, m, o, ka):
    schema = {"type": "object", "required": ["choices"],
              "properties": {"choices": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3}}}
    try:
        d = _json_chat(c, m, o, ka, "Scene: %s\nList 3 different short actions the player could take next." % SAMPLE_SCENE, schema)
    except (ValueError, OllamaError):
        return 0, 1
    return (1 if len(clean_choices(d.get("choices") if isinstance(d, dict) else None, 3)) >= 3 else 0), 1


def t_analyst(c, m, o, ka):
    try:
        d = _json_chat(c, m, o, ka, "Passage: %s\nFill in mood, danger, characters and location." % SAMPLE_SCENE, ANALYSIS_SCHEMA)
    except (ValueError, OllamaError):
        return 0, 1
    return (1 if isinstance(d, dict) and d.get("mood") in MOODS else 0), 1


def t_router(c, m, o, ka):
    cases = [("open the heavy door", "action"), ("what am I carrying", "inventory"),
             ('hello old man, who are you?', "talk")]
    ok = 0
    for text, want in cases:
        try:
            r = c.chat(m, [{"role": "user", "content": ROUTE_PROMPT + text}],
                       options=dict(o, num_predict=4), keep_alive=ka)
        except OllamaError:
            continue
        ok += parse_intent(r["content"]) == want
    return ok, len(cases)


def t_tools(c, m, o, ka):
    msgs = [{"role": "system", "content": "You are a game state tracker. Call a function for every change."},
            {"role": "user", "content": "Passage: You pick up a rusty key from the table.\nCurrent state: carrying: nothing"}]
    try:
        r = c.chat(m, msgs, tools=TOOLS, options=dict(o, num_predict=80), keep_alive=ka)
    except OllamaError:
        return 0, 1
    for call in r["tool_calls"]:
        fn = call.get("function", {})
        if fn.get("name") == "add_item" and "key" in json.dumps(fn.get("arguments", "")).lower():
            return 1, 1
    return 0, 1


ROLE_TESTS = {"narrator": t_narrator, "choices": t_choices, "analyst": t_analyst,
              "router": t_router, "tools": t_tools}


def bench_model(client, cfg, role, model, runs):
    row = {"role": role, "model": model, "present": True, "load": None, "prompt_tps": None,
           "gen_tps": None, "unit": "tok/s", "check": ""}
    client.unload(model)                       # start cold so the load time is real
    time.sleep(0.5)
    ka = cfg["keep_alive"]
    opts = {"temperature": 0, "num_ctx": cfg["num_ctx"]}
    if role == "memory":
        texts = ["A dragon attacks the village.", "A fiery dragon burns the town.",
                 "A recipe for chocolate cake.", "The captain unrolls an old map."]
        rates, ok = [], 0
        for i in range(runs):
            vecs, st = client.embed(model, texts, keep_alive=ka)
            if i == 0:
                row["load"] = st["load"]
            rates.append(len(texts) / max(st["wall"], 1e-6))
            ok += cosine(vecs[0], vecs[1]) > cosine(vecs[0], vecs[2])
        row["gen_tps"], row["unit"], row["check"] = sum(rates) / len(rates), "vec/s", "%d/%d" % (ok, runs)
        return row
    gens, prompts = [], []
    for i in range(runs):
        r = client.chat(model, [{"role": "user", "content": SPEED_PROMPT}],
                        options=dict(opts, num_predict=48), keep_alive=ka)
        st = r["stats"]
        if i == 0:
            row["load"] = st["load"]
        if st["gen_tps"]:
            gens.append(st["gen_tps"])
        if st["prompt_tps"]:
            prompts.append(st["prompt_tps"])
    row["gen_tps"] = sum(gens) / len(gens) if gens else 0.0
    row["prompt_tps"] = sum(prompts) / len(prompts) if prompts else 0.0
    passed = total = 0
    for _ in range(runs):
        p, t = ROLE_TESTS[role](client, model, opts, ka)
        passed, total = passed + p, total + t
    row["check"] = "%d/%d" % (passed, total)
    return row


def estimate_turn(rows, cfg):
    by = {r["role"]: r for r in rows if r.get("gen_tps")}
    plan = [("narrator", 450, LENGTHS[cfg["length"]] * 1.4), ("analyst", 250, 50), ("choices", 250, 40)]
    if cfg["tracking"]:
        plan.append(("tools", 350, 30))
    used = {r for r, _, _ in plan} | ({"memory"} if cfg["recall"] else set())
    swaps = effective_max_loaded(cfg) < len(used)
    total = 0.0
    for role, ptoks, gtoks in plan:
        r = by.get(role) or by.get("narrator")
        if not r:
            continue
        total += gtoks / r["gen_tps"]
        if r.get("prompt_tps"):
            total += ptoks / r["prompt_tps"]
        if swaps and r.get("load"):
            total += r["load"]
    if cfg["recall"] and swaps and "memory" in by:
        total += by["memory"].get("load") or 0
    return total, swaps


def print_bench(rows, cfg, when=""):
    say(rule("Benchmark" + (" (%s)" % when if when else "")))
    say(dim(" %-20s %-8s %6s %8s %9s %6s" % ("model", "role", "load s", "prompt/s", "gen/s", "check")))
    for r in rows:
        label = ROLES[r["role"]][1]
        if not r.get("present"):
            say(" %-20s %-8s %s" % (r["model"][:20], label, red("not installed")))
            continue
        gen = "%.1f%s" % (r["gen_tps"] or 0, "v" if r.get("unit") == "vec/s" else "")
        say(" %-20s %-8s %6.1f %8s %9s %6s" % (
            r["model"][:20], label, r["load"] or 0,
            "%.0f" % r["prompt_tps"] if r.get("prompt_tps") else "-", gen, r["check"]))
    say(dim(" load = cold start from disk; gen/s in tokens/s ('v' = vectors/s); check = role self-test passed"))
    est, swaps = estimate_turn(rows, cfg)
    if est:
        say(" Rough time per turn with current settings: %s%s" % (
            bold("~%.0fs" % est), " (includes model swaps: only %d fit at once)" % effective_max_loaded(cfg) if swaps else ""))


# --------------------------------------------------------------------------- application
class App:
    def __init__(self, cfg):
        self.cfg, self.ai = cfg, AI(cfg)

    # ---- connection / models
    def connect(self, quiet=False):
        try:
            self.ai.reconnect()
            return True
        except OllamaError as e:
            if not quiet:
                say(red("  %s" % e))
                say(dim("  Start it with: sudo systemctl start ollama   (or change the host in x > h)"))
            return False

    def download_hf_model(self, model, on_progress):
        source = HF_GGUF_SOURCES.get(model)
        if not source:
            raise OllamaError("no Hugging Face GGUF mapping for %s; choose an Ollama model name" % model)
        repo, filename = source
        HF_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
        destination = HF_MODEL_DIR / (slug + ".gguf")
        if destination.is_file():
            with open(destination, "rb") as cached:
                valid_cache = cached.read(4) == b"GGUF"
            if valid_cache:
                on_progress({"status": "Using cached Hugging Face file", "completed": 1, "total": 1})
                return destination
            destination.unlink()

        url = "https://huggingface.co/%s/resolve/main/%s?download=true" % (
            urllib.parse.quote(repo, safe="/"), urllib.parse.quote(filename, safe=""))
        headers = {"User-Agent": "StoryForge/%s" % VERSION}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(url, headers=headers)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                if total and shutil.disk_usage(str(HF_MODEL_DIR)).free < total:
                    raise OllamaError("not enough free disk space for %s" % filename)
                completed = 0
                with open(partial, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        completed += len(chunk)
                        on_progress({"status": "Downloading %s" % filename,
                                     "completed": completed, "total": total})
            if total and completed != total:
                raise OllamaError("incomplete Hugging Face download (%d of %d bytes)" % (completed, total))
            os.replace(partial, destination)
            with open(destination, "rb") as downloaded:
                if downloaded.read(4) != b"GGUF":
                    destination.unlink()
                    raise OllamaError("Hugging Face response was not a GGUF file")
        except OllamaError:
            try:
                partial.unlink()
            except OSError:
                pass
            raise
        except urllib.error.HTTPError as error:
            try:
                partial.unlink()
            except OSError:
                pass
            if error.code == 401:
                message = "Hugging Face denied access; this file may require HF_TOKEN"
            else:
                message = "Hugging Face download failed (HTTP %d)" % error.code
            raise OllamaError(message, error.code) from None
        except (urllib.error.URLError, OSError) as error:
            try:
                partial.unlink()
            except OSError:
                pass
            raise OllamaError("Hugging Face download failed (%s)" % getattr(error, "reason", error)) from None
        return destination

    def pull_models(self, models, source=None):
        if source is None:
            say("  Model source: o) Ollama library  h) Hugging Face GGUF")
            selected = ask("  Pull source [o/h]> ").lower()
            if selected in ("o", "ollama"):
                source = "ollama"
            elif selected in ("h", "huggingface", "hf"):
                source = "huggingface"
            else:
                say(dim("  Pull cancelled; choose o or h."))
                return
        ollama_online = True
        if source == "huggingface":
            try:
                self.ai.client.version()
            except OllamaError:
                ollama_online = False
                say(dim("  Ollama is unreachable. Hugging Face files will be cached now; retry this pull "
                        "to import them after Ollama is online."))
        for m in dict.fromkeys(models):
            say("  Pulling %s from %s ..." % (m, "Hugging Face" if source == "huggingface" else "Ollama"))
            last = {"s": ""}

            def prog(o):
                st = o.get("status", "")
                if o.get("total") and o.get("completed"):
                    line = "  %s %d%%" % (st[:30], 100 * o["completed"] // o["total"])
                else:
                    line = "  " + st[:40]
                if sys.stdout.isatty():
                    sys.stdout.write("\r\033[K" + line)
                    sys.stdout.flush()
                elif st != last["s"]:
                    say(line)
                last["s"] = st

            try:
                if source == "huggingface":
                    model_file = self.download_hf_model(m, prog)
                    if ollama_online:
                        digest = self.ai.client.push_blob(model_file, prog)
                        self.ai.client.create(m, {model_file.name: digest}, prog)
                    else:
                        say(dim("  Cached %s; retry when Ollama is reachable." % model_file.name))
                else:
                    self.ai.client.pull(m, prog)
                if sys.stdout.isatty():
                    say()
            except (OllamaError, OSError) as e:
                say(red("\n  Could not pull/import %s: %s" % (m, e)))
        self.connect(quiet=True)

    def ensure_ready(self):
        if not self.ai.online and not self.connect():
            return False
        missing = self.ai.missing()
        if not self.ai.has("narrator"):
            say(red("  The narrator model %s is not installed." % self.cfg["models"]["narrator"]))
            say("  Pull it now? (y/n) ")
            if getkey() == "y":
                self.pull_models([self.cfg["models"]["narrator"]])
            return self.ai.has("narrator")
        if missing:
            say(dim("  Running without: %s  (x > 1 > p pulls them)" % ", ".join(
                "%s (%s)" % (ROLES[r][1].lower(), m) for r, m in missing)))
        return True

    # ---- saves
    def save_state(self, s):
        s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            path = SAVE_DIR / (s["id"] + ".json")
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(s, f)
            os.replace(tmp, path)
        except OSError as e:
            say(dim("  (could not save: %s)" % e))

    def list_saves(self):
        out = []
        for p in sorted(SAVE_DIR.glob("*.json")) if SAVE_DIR.exists() else []:
            try:
                with open(p) as f:
                    s = json.load(f)
                if isinstance(s, dict) and "turns" in s and "id" in s:
                    out.append(s)
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda s: s.get("updated", ""), reverse=True)

    def export(self, s):
        try:
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            path = EXPORT_DIR / ("%s-%s.md" % (re.sub(r"[^A-Za-z0-9]+", "-", s["preset"]).strip("-").lower(), s["id"]))
            lines = ["# %s story" % s["preset"], "", "*%s*" % s["setting"], "",
                     "You are %s. Goal: %s" % (s["role"], s["objective"]), ""]
            for t in s["turns"]:
                if t["action"] not in ("(the story begins)", "(the end)"):
                    lines += ["> %s" % t["action"], ""]
                lines += [t["scene"], ""]
            path.write_text("\n".join(lines), encoding="utf-8")
            say(green("  Exported to %s" % path))
        except OSError as e:
            say(red("  Could not export: %s" % e))

    # ---- starting stories
    def start(self, preset):
        if not self.ensure_ready():
            return
        Game(self, new_state(preset, self.cfg["performance_mode"])).play()

    def custom_world(self):
        if not self.ensure_ready():
            return
        say()
        idea = ask("  Describe your world in a sentence or two> ")
        if not idea:
            return
        p = dict(name="Custom", tone="adventurous", setting=idea, role="a traveler",
                 location="the start of your journey", objective="find out what is going on", items=[])
        msgs = [{"role": "user", "content":
                 'Turn this idea into a story setup: "%s". Fill in: name (1-2 words), tone (a few words), '
                 "setting (one or two sentences), role (who the player is, e.g. \"a lighthouse keeper\"), "
                 "location (where the story starts), objective (the player's first goal), items (2-3 "
                 "starting items)." % idea}]
        try:
            res = self.ai.chat("narrator", msgs, "world", fmt=SETUP_SCHEMA,
                               options={"temperature": 0.7, "num_predict": 220})
            d = json.loads(res["content"])
            for k in ("name", "tone", "setting", "role", "location", "objective"):
                v = sstr(d.get(k), 160)
                if v:
                    p[k] = v
            items = [sstr(i, 30) for i in d.get("items", []) if sstr(i, 30)][:4]
            p["items"] = items or p["items"]
        except (OllamaError, Unavailable, ValueError, AttributeError, TypeError):
            say(dim("  (could not expand the idea - using it as it is)"))
        say(dim("  World: %s | You are %s | Goal: %s" % (p["name"], p["role"], p["objective"])))
        Game(self, new_state(p, self.cfg["performance_mode"])).play()

    def load_menu(self):
        saves = self.list_saves()
        if not saves:
            say(dim("  No saved stories yet."))
            return
        while True:
            say()
            say(rule("Continue a story"))
            shown = saves[:9]
            for i, s in enumerate(shown, 1):
                timeline = s.get("world_state", {}).get("timeline", {}).get("name", "A")
                say("  %d) %-10s %-6s turn %-3d %s%s" % (
                    i, s["preset"][:10], timeline[:6], len(s["turns"]), s.get("updated", "")[:16],
                    " (finished)" if s.get("over") else ""))
            say(dim("  1-%d load | d delete one | q back" % len(shown)))
            k = getkey()
            if k.isdigit() and 1 <= int(k) <= len(shown):
                s = shown[int(k) - 1]
                for key, val in new_state({"name": "", "tone": "", "setting": "", "role": "", "location": "",
                                           "objective": "", "items": []}, self.cfg["performance_mode"]).items():
                    s.setdefault(key, val)
                if self.ensure_ready():
                    Game(self, s).play()
                return
            if k == "d":
                say("  Delete which one? (1-%d) " % len(shown))
                n = getkey()
                if n.isdigit() and 1 <= int(n) <= len(shown):
                    try:
                        (SAVE_DIR / (shown[int(n) - 1]["id"] + ".json")).unlink()
                    except OSError:
                        pass
                    saves = self.list_saves()
                    if not saves:
                        return
            elif k in ("q", "esc", "\n"):
                return

    # ---- settings
    def models_menu(self):
        c = self.cfg
        while True:
            say()
            say(rule("Models per role"))
            for i, (role, (_d, label, desc)) in enumerate(ROLES.items(), 1):
                m = c["models"][role]
                flag = green("ok     ") if norm(m) in self.ai.installed else red("missing")
                say(" %d) %-9s %-20s %s %s" % (i, label, m, flag, dim(desc)))
            say(dim(" 1-6 change a model | p pull missing (Ollama/Hugging Face) | d defaults | x back"))
            k = getkey()
            if k.isdigit() and 1 <= int(k) <= len(ROLES):
                role = list(ROLES)[int(k) - 1]
                name = ask("  Model for %s (Enter keeps %s)> " % (ROLES[role][1], c["models"][role]))
                if name:
                    c["models"][role] = name
                    self.connect(quiet=True)
            elif k == "p":
                miss = [m for _r, m in self.ai.missing()]
                if miss:
                    self.pull_models(miss)
                else:
                    say(dim("  Nothing missing."))
            elif k == "d":
                apply_performance_mode(c, c["performance_mode"])
                self.connect(quiet=True)
            elif k in ("x", "q", "esc", "\n"):
                break
            save_config(c)

    def performance_menu(self):
        say()
        say(rule("Performance mode"))
        for key, mode in (("l", "low"), ("m", "medium"), ("h", "high")):
            profile = PERFORMANCE_MODES[mode]
            say("  %s) %-7s %-24s context %d" % (
                key, profile["label"], profile["ram"], profile["num_ctx"]))
        say(dim("  Medium and High enable persistent events, character memories, relationships, dialogue and trade."))
        key = getkey()
        mode = {"l": "low", "m": "medium", "h": "high"}.get(key)
        if mode:
            apply_performance_mode(self.cfg, mode)
            save_config(self.cfg)
            self.connect(quiet=True)
            say(green("  Selected %s mode." % PERFORMANCE_MODES[mode]["label"]))
            if self.ai.missing():
                say(dim("  Missing models can be pulled in Settings > Models per role > p."))

    def settings_menu(self):
        c = self.cfg
        while True:
            say()
            say(rule("Settings"))
            if self.ai.online:
                try:
                    loaded = ", ".join("%s (%.1f GB)" % (m.get("name"), m.get("size", 0) / 1e9)
                                       for m in self.ai.client.ps()) or "nothing"
                except OllamaError:
                    loaded = "?"
                say(dim(" Ollama %s | in memory now: %s" % (self.ai.version, loaded)))
            else:
                say(red(" Ollama offline (%s)" % self.ai.client.base))
            ml = c["max_loaded"]
            rows = [
                ("p", "Performance mode", PERFORMANCE_MODES[c["performance_mode"]]["label"]),
                ("1", "Models per role", "%d of %d installed" % (len(ROLES) - len(self.ai.missing()), len(ROLES))),
                ("2", "Models in memory at once", "auto (%d)" % auto_max_loaded() if not ml else str(ml)),
                ("3", "Keep models loaded for", c["keep_alive"]),
                ("4", "Scene length", "%s (~%d words)" % (c["length"], LENGTHS[c["length"]])),
                ("5", "Creativity (temperature)", "%.1f" % c["temperature"]),
                ("6", "Choices per turn", str(c["choices"])),
                ("7", "Memory recall (embeddings)", "top %d" % c["recall"] if c["recall"] else "off"),
                ("8", "State tracking (tool calls)", onoff(c["tracking"])),
                ("9", "Stream text as it is written", onoff(c["stream"])),
                ("0", "Show timings", onoff(c["timing"])),
                ("h", "Ollama host", c["host"]),
                ("b", "Benchmark models", ""),
                ("r", "Reset to defaults", ""),
                ("x", "Back", ""),
            ]
            for key, label, val in rows:
                say(" %s) %-30s %s" % (key, label, val))
            say(dim(" Fewer models in memory = less RAM but slower turns; use b to measure."))
            k = getkey()
            if k == "p":
                self.performance_menu()
            elif k == "1":
                self.models_menu()
            elif k == "2":
                c["max_loaded"] = cycle([None, 1, 2, 3, 6], c["max_loaded"])
            elif k == "3":
                c["keep_alive"] = cycle(KEEP_ALIVES, c["keep_alive"])
            elif k == "4":
                c["length"] = cycle(list(LENGTHS), c["length"])
            elif k == "5":
                c["temperature"] = cycle([0.4, 0.6, 0.8, 1.0], c["temperature"])
            elif k == "6":
                c["choices"] = cycle([2, 3, 4], c["choices"])
            elif k == "7":
                c["recall"] = cycle([0, 2, 3, 5], c["recall"])
            elif k == "8":
                c["tracking"] = not c["tracking"]
            elif k == "9":
                c["stream"] = not c["stream"]
            elif k == "0":
                c["timing"] = not c["timing"]
            elif k == "h":
                host = ask("  Ollama host (Enter keeps %s)> " % c["host"])
                if host:
                    c["host"] = host
                    self.connect()
            elif k == "b":
                self.benchmark_menu()
            elif k == "r":
                self.cfg.clear()
                self.cfg.update(json.loads(json.dumps(DEFAULTS)))
                c = self.cfg
                self.connect(quiet=True)
            elif k in ("x", "q", "esc", "\n"):
                save_config(c)
                return
            save_config(c)

    # ---- benchmark
    def run_benchmark(self, runs):
        if not self.ai.online and not self.connect():
            return
        rows = []
        for role, (_d, label, _desc) in ROLES.items():
            model = self.cfg["models"][role]
            if norm(model) not in self.ai.installed:
                rows.append({"role": role, "model": model, "present": False})
                say(" %-20s %s" % (model[:20], red("not installed - skipped")))
                continue
            say(dim(" testing %s (%s) ..." % (model, label.lower())))
            st = Status("benchmark %s" % model)
            try:
                row = bench_model(self.ai.client, self.cfg, role, model, runs)
            except OllamaError as e:
                st.stop()
                say(red("  failed: %s" % e))
                rows.append({"role": role, "model": model, "present": True, "load": 0, "gen_tps": 0,
                             "unit": "tok/s", "check": "error"})
                continue
            finally:
                st.stop()
            self.ai.client.unload(model)               # one model at a time, like the game
            rows.append(row)
        say()
        print_bench(rows, self.cfg)
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(BENCH_FILE, "w") as f:
                json.dump({"when": time.strftime("%Y-%m-%d %H:%M"), "runs": runs, "rows": rows}, f)
        except OSError:
            pass

    def benchmark_menu(self):
        say()
        try:
            with open(BENCH_FILE) as f:
                last = json.load(f)
            print_bench(last["rows"], self.cfg, "last run %s" % last["when"])
        except (OSError, ValueError, KeyError):
            say(dim("  No benchmark yet."))
        say()
        say("  Run: q) quick (1 pass, ~1-3 min)   f) full (3 passes)   x) back")
        say(dim("  Each model is unloaded after its test, so load times are cold-start times."))
        k = getkey()
        if k in ("q", "f"):
            self.run_benchmark(1 if k == "q" else 3)

    # ---- main menu
    def status_line(self):
        if not self.ai.online:
            return red(" Ollama offline (%s) - press x to change the host" % self.ai.client.base)
        miss = self.ai.missing()
        return dim(" Ollama %s | %s mode | models ready: %d/%d | %d in memory at once" % (
            self.ai.version, PERFORMANCE_MODES[self.cfg["performance_mode"]]["label"],
            len(ROLES) - len(miss), len(ROLES), effective_max_loaded(self.cfg)))

    def run(self):
        self.connect(quiet=True)
        while True:
            say()
            say(rule("STORYFORGE"))
            say(self.status_line())
            say(" Pick a world:")
            for i, p in enumerate(PRESETS, 1):
                say("  %d) %-11s %s" % (i, p["name"], dim(p["blurb"])))
            say("  n) Your own world       c) Continue a story")
            say("  p) Performance mode     b) Benchmark models     x) Settings       q) Quit")
            k = getkey()
            if k.isdigit() and 1 <= int(k) <= len(PRESETS):
                self.start(PRESETS[int(k) - 1])
            elif k == "n":
                self.custom_world()
            elif k == "c":
                self.load_menu()
            elif k == "b":
                self.benchmark_menu()
            elif k == "p":
                self.performance_menu()
            elif k == "x":
                self.settings_menu()
            elif k in ("q", "esc"):
                return


def main():
    ap = argparse.ArgumentParser(description="Interactive stories with tiny local Ollama models.")
    ap.add_argument("--host", help="Ollama host (default: $OLLAMA_HOST or 127.0.0.1:11434)")
    ap.add_argument("--benchmark", action="store_true", help="benchmark the models and exit")
    ap.add_argument("--full", action="store_true", help="with --benchmark: 3 passes instead of 1")
    ap.add_argument("--version", action="version", version="StoryForge " + VERSION)
    args = ap.parse_args()
    cfg = load_config()
    if args.host:
        cfg["host"] = args.host
    app = App(cfg)
    try:
        if args.benchmark:
            if app.connect():
                app.run_benchmark(3 if args.full else 1)
            return
        app.run()
    except (KeyboardInterrupt, EOFError):
        pass
    say()
    say(dim("  Your stories are saved in %s" % SAVE_DIR))


if __name__ == "__main__":
    main()
