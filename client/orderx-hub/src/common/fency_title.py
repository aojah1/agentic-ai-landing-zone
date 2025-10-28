# super_fancy_title.py
# Ultra-fancy animated title with rainbow gradient, sparkles, and a shimmering sweep.
# No external libraries. Works best in a truecolor terminal.

import sys, time, math, shutil, os
from itertools import cycle

TITLE = "Sales Order Automation Agent"

# ---------- ANSI helpers ----------
RESET = "\033[0m"
HIDE = "\033[?25l"
SHOW = "\033[?25h"

def rgb(r, g, b): return f"\033[38;2;{r};{g};{b}m"
def bg(r, g, b):  return f"\033[48;2;{r};{g};{b}m"

def hsv_to_rgb(h, s=1.0, v=1.0):
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = int(255 * v * (1.0 - s))
    q = int(255 * v * (1.0 - f * s))
    t = int(255 * v * (1.0 - (1.0 - f) * s))
    v = int(255 * v)
    i = i % 6
    if i == 0: return (v, t, p)
    if i == 1: return (q, v, p)
    if i == 2: return (p, v, t)
    if i == 3: return (p, q, v)
    if i == 4: return (t, p, v)
    return (v, p, q)

def gradient_text(text, base_hue=0.0, sat=0.9, val=1.0):
    n = max(1, len(text) - 1)
    out = []
    for i, ch in enumerate(text):
        h = (base_hue + i / max(10, n)) % 1.0
        r, g, b = hsv_to_rgb(h, sat, val)
        out.append(f"{rgb(r,g,b)}{ch}{RESET}")
    return "".join(out)

def gradient_line(width, base_hue, chars=("═", "═")):
    # Left→Right hue sweep for borders
    left_cap, right_cap = chars
    seg = []
    for i in range(width):
        h = (base_hue + i / max(2, width)) % 1.0
        r, g, b = hsv_to_rgb(h, 0.9, 1.0)
        seg.append(f"{rgb(r,g,b)}═{RESET}")
    return f"{left_cap}{''.join(seg)}{right_cap}"

# ---------- Sparkle logic ----------
STAR_SET = ["·", "•", "⋆", "✦", "✧", "✩"]
STAR_COLORS = [0.15, 0.35, 0.55, 0.75, 0.9]  # hues

def starfield(width, base_hue, density=0.08):
    from random import random, choice
    s = []
    for i in range(width):
        if random() < density:
            ch = choice(STAR_SET)
            h = (base_hue + choice(STAR_COLORS)) % 1.0
            r,g,b = hsv_to_rgb(h, 0.6, 0.9)
            s.append(f"{rgb(r,g,b)}{ch}{RESET}")
        else:
            s.append(" ")
    return "".join(s)

# ---------- Box render ----------
def render_frame(title, base_hue=0.0, shimmer_pos=0, width=None):
    cols = shutil.get_terminal_size(fallback=(100, 24)).columns if width is None else width
    inner_w = max(len(title) + 8, min(cols - 6, 120))
    # Build borders
    top = gradient_line(inner_w, base_hue, ("╔", "╗"))
    bot = gradient_line(inner_w, (base_hue + 0.1) % 1.0, ("╚", "╝"))

    # Decorative header/footer lines
    deco = starfield(inner_w, (base_hue + 0.2) % 1.0, density=0.12)
    deco_line = f"║{deco}║"

    # Title line with gradient + shimmer sweep (bright mask)
    grad = gradient_text(title, (base_hue + 0.65) % 1.0, 0.95, 1.0)

    # Pad & center
    left_pad = (inner_w - len(title)) // 2
    right_pad = inner_w - len(title) - left_pad
    line_chars = list((" " * left_pad) + grad + (" " * right_pad))

    # Apply shimmer: a moving bright diagonal streak over the text area
    streak_width = 6
    for i in range(len(title)):
        idx = left_pad + i
        # Distance from shimmer center
        d = abs(idx - shimmer_pos)
        if d <= streak_width:
            # boost brightness
            h = (base_hue + 0.65 + i / max(10, len(title))) % 1.0
            r, g, b = hsv_to_rgb(h, 0.6, 1.0)
            # subtle glow background
            br, bgc, bb = (min(255, r + 30), min(255, g + 30), min(255, b + 30))
            line_chars[idx] = f"{bg(br, bgc, bb)}{rgb(0,0,0)}{title[i]}{RESET}"

    mid = f"║{''.join(line_chars)}║"

    # Soft drop shadow (dim underline)
    shadow = f"\033[2m  {'▁' * inner_w}{RESET}"

    return "\n".join([top, deco_line, mid, deco_line, bot, shadow])

# ---------- Main animation ----------
def animate(title=TITLE, fps=24, duration=None):
    # If not a TTY or NO_COLOR set, we still print something decent
    is_tty = sys.stdout.isatty()
    try:
        print(HIDE, end="")
        t0 = time.time()
        phase = 0.0
        while True:
            cols = shutil.get_terminal_size(fallback=(100, 24)).columns
            inner_w = max(len(title) + 8, min(cols - 6, 120))
            shimmer_pos = int((math.sin(time.time() * 2.0) + 1) * 0.5 * (inner_w - 1))
            frame = render_frame(title, base_hue=phase, shimmer_pos=shimmer_pos)
            # Move cursor to top-left and clear screen region
            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(frame + "\n")
            sys.stdout.flush()
            phase = (phase + 0.007) % 1.0
            if duration and (time.time() - t0) >= duration:
                break
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        print(SHOW + RESET, end="")

if __name__ == "__main__":
    # Options via env:
    #   TITLE_TEXT        -> override title
    #   FANCY_DURATION    -> seconds to run (float). Omit for infinite until Ctrl-C
    #   FANCY_FPS         -> frames per second (int)
    title = os.getenv("TITLE_TEXT", TITLE)
    dur = os.getenv("FANCY_DURATION", 2)
    fps = int(os.getenv("FANCY_FPS", "24"))
    duration = float(dur) if dur else None
    animate(title, fps=fps, duration=duration)
