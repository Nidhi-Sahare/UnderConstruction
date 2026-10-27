r"""
apply_hero.py - installs the new Under-construction hero into
src/construction_dashboard.py.

Run from the project root (the folder that contains src\):
    .\.venv\Scripts\python.exe apply_hero.py

What it does (nothing else in the file is touched):
  1. Backs up src/construction_dashboard.py to construction_dashboard.py.bak
  2. Replaces everything from <!-- UC_STEP114_START --> to
     <!-- UC_STEP114_END --> with the contents of hero_block.html
  3. Replaces both <div class="brand-mark">...</div> logos with the
     lavender hard-hat SVG.
Safe to run again: it finds the same markers and replaces them again.
"""
import pathlib
import re
import shutil
import sys

TARGET = pathlib.Path("src") / "construction_dashboard.py"
BLOCK = pathlib.Path("hero_block.html")
START = "<!-- UC_STEP114_START -->"
END = "<!-- UC_STEP114_END -->"

HELMET = (
    '<div class="brand-mark" aria-hidden="true">'
    '<svg viewBox="0 0 50 40" xmlns="http://www.w3.org/2000/svg" focusable="false">'
    '<path fill="currentColor" d="M10 30C10 17 16.5 9 25 9s15 8 15 21z"/>'
    '<path fill="currentColor" d="M20.5 10.5C20.5 6 22.3 3.5 25 3.5s4.5 2.5 4.5 7z"/>'
    '<path fill="currentColor" d="M3 29.5h44a2.5 2.5 0 0 1 2.5 2.5v1.5a3 3 0 0 1-3 3h-43a3 3 0 0 1-3-3V32a2.5 2.5 0 0 1 2.5-2.5z"/>'
    '<path d="M20.5 10.5V30M29.5 10.5V30" stroke="#15141c" stroke-opacity=".45" stroke-width="1.4" fill="none"/>'
    '<path d="M14 29C14 21 17 15 21 12" stroke="#ffffff" stroke-opacity=".28" stroke-width="1.6" fill="none" stroke-linecap="round"/>'
    '</svg></div>'
)


def fail(msg):
    print("ERROR:", msg)
    sys.exit(1)


if not TARGET.exists():
    fail("cannot find %s - run this from the project root." % TARGET)
if not BLOCK.exists():
    fail("cannot find %s next to this script." % BLOCK)

raw = TARGET.read_bytes()
try:
    src = raw.decode("utf-8")
except UnicodeDecodeError:
    fail("the dashboard file is not UTF-8; no changes made.")

nl = "\r\n" if "\r\n" in src else "\n"
block = BLOCK.read_text(encoding="utf-8").replace("\r\n", "\n")

if '"""' in block:
    fail("hero_block.html contains a triple quote, which would end the HTML string.")

i = src.find(START)
j = src.find(END)
if i < 0 or j < 0 or j < i:
    fail("could not find the %s ... %s markers." % (START, END))

triples_before = src.count('"""')

new_src = src[:i] + block.replace("\n", nl) + src[j + len(END):]
new_src, n_logo = re.subn(
    r'<div class="brand-mark"[^>]*>.*?</div>',
    lambda m: HELMET,
    new_src,
    flags=re.S,
)

if new_src.count('"""') != triples_before:
    fail("triple-quote count changed; aborting without writing.")

shutil.copyfile(TARGET, TARGET.with_suffix(".py.bak"))
TARGET.write_bytes(new_src.encode("utf-8"))

print("Hero block replaced.")
print("Logos replaced:", n_logo, "(expected 2)")
print("Backup written to", TARGET.with_suffix(".py.bak"))
print("Now run:  .\\.venv\\Scripts\\python.exe src\\construction_dashboard.py")
print("and hard-refresh the browser with Ctrl+F5.")
