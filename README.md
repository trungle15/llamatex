# llamatex

Convert natural language descriptions to LaTeX equations in your terminal, in under a second!

## Features

`llamatex` offers two primary modes of operation:
1. Single Instruction Mode: Generate LaTeX for a single equation description.
2. Clipboard Mode: Send LaTeX code from highlighted text into your clipboard.
3. Markdown File Editing Mode: Automatically replace natural language descriptions with LaTeX equations in a markdown file.

For equations that can be rendered in a single line, smoltex opens a new window displaying the rendered equation.

## Installation

Since `llmatex` is a CLI application built and run from Python, it is recommended that you install it via `pipx` to isolate the program from your global interpreter. You can install `pipx` via homebrew.

```bash
pipx install llamatex
```

For local generation support:

```bash
pip install llamatex[local]
```

Alternatively, you can build from source using `poetry`.

```bash
poetry install
```

## Usage

### Single Instruction mode

```bash
llamatex render -p "equation for the area of a circle"
```

Output:

```
Latex string: A = \pi r^2

>> completion time: 856.4 ms
```

### Clipboard mode

> [!WARNING]
> `llamatex` clipboard mode currently only supports MacOS

`llamatex` uses `pynput` to listen to your inputs and control the keyboard. To allow clipboard mode, allow your terminal/IDE to be added to Assessibility in System Settings.

Run `llamatex` in clipboard mode using the following command:

```bash
llamatex run
```

### Markdown File Editing Mode
1. Create a markdown file (e.g., `equations.md`) with placeholders:

```markdown
# Important Equations

1. Area of a circle:
{"area of a circle"}

2. Einstein's mass-energy equivalence:
{"einstein's famous equation"}
```
2. Run `llmatex`:

```bash
llamatex render -f equations.md
```

3. The file will be updated with LaTeX equations:

```markdown
# Important Equations

1. Area of a circle:
$A = \pi r^2$

2. Einstein's mass-energy equivalence:
$E = mc^2$
```

Use the hotkey `Cmd+Shift+L` (on macOS) to process the selected text. Press `esc` to exit the program.

## Configuration

### API Setup

1. Get a free API key from Groq console.
2. Set the environment variable

```bash
export GROQ_API_KEY=your-api-key-here
```

Available models:
- llama3: `llama3-8b`, `llama3-70b`, `llama3.1-8b`, `llama3.1-70b`
- gemma: `gemma-7b`, `gemma2-9b`
- mixtral: `mixtral-8x7b`
- Phi3 (local only): `phi-3-mini`

### Server selection
Choose between `groq` (default) or `llama-cpp` (for local generation) using the `-s` or `--server` option:

```bash
llamatex render -p "pythagorean theorem" -s llama-cpp
```







