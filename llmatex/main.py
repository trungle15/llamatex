import os
import re
import time
import rich
from rich.markdown import Markdown
from rich.panel import Panel
import argparse
import pyperclip
from pynput import keyboard
from pynput.keyboard import Key, Controller

from typing import *
from groq import Groq

from .prompts import LATEX_PROMPT
from .render import show_latex_equation

try:
    import torch
    import transformers
    from llama_cpp import Llama, llama_tokenizer
    LOCAL_MODEL_AVAILABLE = True
except ImportError as e:
    LOCAL_MODEL_AVAILABLE = False


if LOCAL_MODEL_AVAILABLE:
    transformers.logging.set_verbosity_error()

MAX_PROMPT_LENGTH = 500 # characters
DEFAULT_MODEL = "llama3.1-70b"
AVAILABLE_MODELS_GROQ = [
    "llama3-8b",
    "llama3-70b",
    "llama3.1-8b",
    "llama3.1-70b",
    "mixtral-8x7b",
    "gemma-7b",
    "gemma2-9b",
]
AVAILABLE_MODELS_LLAMA = [
    "llama3-8b",
    "gemma2-9b",
    "phi-3-mini",
]
MODEL_KEYS_GROQ = {"llama3-8b": "llama3-8b-8192", 
                   "llama3-70b": "llama3-70b-8192", 
                   "llama3.1-8b": "llama-3.1-8b-instant",
                   "llama3.1-70b": "llama-3.1-70b-versatile",
                   "mixtral-8x7b": "mixtral-8x7b-32768", 
                   "gemma-7b": "gemma-7b-it", 
                   "gemma2-9b": "gemma2-9b-it"}
MODEL_KEYS_LLAMA = {"llama3-8b": "bartowski/Llama-3-Instruct-8B-SPPO-Iter3-GGUF", 
                    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct", 
                    "gemma2-9b": "bartowski/gemma-2-9b-it-GGUF"}
FILE_KEYS_LLAMA = {"llama3-8b": "Llama-3-Instruct-8B-SPPO-Iter3-Q4_K_M.gguf", 
                   "phi-3-mini": "Phi-3-mini-4k-instruct-q4.gguf", 
                   "gemma2-9b": "gemma-2-9b-it-Q4_K_L.gguf"}
TOKENIZER_KEYS_LLAMA = {"llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct", 
                        "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct", 
                        "gemma2-9b": "google/gemma-2-27b-it"}
AVAILABLE_SERVERS=[
    "groq",
    "llama-cpp"
]
DEFAULT_SERVER = "groq"

stop_program = False
controller = Controller()
original_clipboard = None

def verbose_print(message, verbose=False):
    if verbose:
        print(message)


def create_llm_client(server_type: Optional[str] = DEFAULT_SERVER, model:str = DEFAULT_MODEL, verbose: bool = False):
    if server_type == "groq":
        try:
            llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            rich.print(f">> [bold green]Groq LLM client created successfully.[/]")
            return llm
        except Exception as e:
            rich.print(f">> [bold red]ClientError: cannot create the Groq LLM client.[/]")
            exit(1)
    elif server_type == "llama-cpp":
        if not LOCAL_MODEL_AVAILABLE:
            rich.print(">> [bold red]ServerError: cannot use llama-cpp server, make sure to install using `pip install 'smoltex[local]'`")
            exit(1)
        try:
            llm = Llama.from_pretrained(
                repo_id=MODEL_KEYS_LLAMA[model],
                filename=FILE_KEYS_LLAMA[model],
                tokenizer = llama_tokenizer.LlamaHFTokenizer.from_pretrained(TOKENIZER_KEYS_LLAMA[model]),
                n_gpu_layers = -1,
                n_ctx = 1024,
                n_batch_size = 512,
                verbose = False,
            )
            rich.print(f">> [bold green]Llama.CPP client created successfully.[/]")
            return llm
        except:
            rich.print(f">> [bold red]ClientError: cannot gcreate the Llama.CPP client.[/]")
            exit(1)
    else:
        rich.print(f">> [bold red]ValueError: Invalid server_type {server_type}.[/]")
        exit(1)


def get_latex_response(
        llm: Any,
        user_input: str,
        model_name: Optional[str] = DEFAULT_MODEL,
        verbose: bool = False
    ) -> str:

    prompt = LATEX_PROMPT.format(user_input=user_input)
    messages = [dict(role="user", content=prompt)]
    
    verbose_print(f">> Sending prompt to model: {model_name}", verbose)
    
    output: str = ""
    if isinstance(llm, Groq):
        if model_name not in AVAILABLE_MODELS_GROQ:
            rich.print(f">> [bold red]ModelError: the model name {model_name} is not available.[/]")
            exit(1)
            
        chat_completion = llm.chat.completions.create(
            messages=messages,
            model=MODEL_KEYS_GROQ[model_name],
        )
        output = chat_completion.choices[0].message.content.strip()

    elif isinstance(llm, Llama):
        chat_completion = llm.create_chat_completion(messages=messages)
        output = chat_completion["choices"][0]["message"]["content"].strip()

    verbose_print(">> Response received from model", verbose)

    if not output.startswith("latex:"):
        rich.print(f">> [bold red]OutputError: invalid output encountered, try again.[/]")
        exit(1)

    output = output.replace("latex:", "").strip()
    if output.startswith("$$") and output.endswith("$$"):
        output = output.replace("$$", "").strip()
    if output.startswith("$") and output.endswith("$"):
        output = output.replace("$", "").strip()
    if output.startswith('"') and output.endswith('"'):
        n = len(output)
        output = output[1:n-1]
    
    verbose_print(f"\nLatex string: {output}", verbose)
    return output


def process_markdown_latex(
    input_file: str,
    llm: Any,
    get_latex_response: Callable[[Any, str, str, bool], str],
    model_name: Optional[str],
    start_delimiter: Optional[str],
    end_delimiter: Optional[str],
    verbose: bool = False,
) -> None:
    start_delimiter_escaped = re.escape(start_delimiter)
    end_delimiter_escaped = re.escape(end_delimiter)
    
    pattern = rf'{start_delimiter_escaped}\s*(.*?)\s*{end_delimiter_escaped}'

    with open(input_file, 'r') as file:
        content = file.read()
    
    def replace_with_latex(match):
        text = match.group(1)
        latex = get_latex_response(llm, text, model_name=model_name, verbose=verbose)
        return f"${latex}$"
    
    verbose_print(f">> Processing file: {input_file}", verbose)
    processed_content = re.sub(pattern, replace_with_latex, content)
    
    with open(input_file, 'w') as file:
        file.write(processed_content)
    
    rich.print(f"\n[bold green]File processed successfully! and saved to [/] [bold blue]{input_file}[/]")

def listen_clipboard(llm: Any, model_name: Optional[str]):
    HOTKEY = {Key.cmd, Key.shift, keyboard.KeyCode.from_char('l')}  # Command+Shift+L
    current_keys = set()
    last_execution_time = 0
    DEBOUNCE_TIME = 3
    keyboard_controller = Controller()

    def copy_and_convert():
        nonlocal last_execution_time
        current_time = time.time()
        
        if current_time - last_execution_time < DEBOUNCE_TIME:
            return

        last_execution_time = current_time

        # Simulate Command+C to copy selected text
        with keyboard_controller.pressed(Key.cmd):
            keyboard_controller.tap('c')
        
        time.sleep(0.5)  # Small delay to ensure clipboard is updated
        
        text = pyperclip.paste()
        
        if not text:
            rich.print(">> [bold yellow]No text copied[/]")
            return

        rich.print(f">> Processing clipboard text: {text[:50]}...")
        
        try:
            latex_response = get_latex_response(llm, text, model_name)
            pyperclip.copy(latex_response)
            
            latex_markdown = Markdown(f"```latex\n{latex_response}\n```")
            latex_panel = Panel(
                latex_markdown,
                title="LaTex Response",
                expand=False,
                border_style="green"
            )
            
            rich.print(f">> [bold green]Clipboard updated with LaTeX response[/]:")
            rich.print(latex_panel)
            
        except Exception as e:
            rich.print(f">> [bold red]Error processing clipboard text: {e}[/]")

    def on_press(key):
        current_keys.add(key)
        if all(k in current_keys for k in HOTKEY):
            copy_and_convert()
        if key == Key.esc:
            rich.print(">> [bold yellow]Program stopped by user.[/]")
            return False

    def on_release(key):
        try:
            current_keys.remove(key)
        except KeyError:
            pass

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        rich.print(">> Listening for clipboard changes. Press Cmd+Shift+L to process, or Esc to quit.")
        listener.join()


def list_models():
    rich.print("[bold green]Available models:[/]")
    rich.print("\n[bold blue]Groq models:[/]")
    for model in AVAILABLE_MODELS_GROQ:
        rich.print(f"  - {model}")
    
    if not LOCAL_MODEL_AVAILABLE:
        rich.print("\nWarning: Below models are not available on this device.")
        print('Install using `pip install "smoltex[local]"`')
        rich.print("\n[bold blue]Llama.cpp models:[/]")
        for model in AVAILABLE_MODELS_LLAMA:
            print(f"  - {model}")


def create_parser():
    parser = argparse.ArgumentParser(description="llmatex: convert natural language to latex equations.")
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # run mode
    run_parser = subparsers.add_parser('run', help="Run in clipboard mode")
    run_parser.add_argument("-m", "--model_name", type=str, choices=AVAILABLE_MODELS_GROQ, help="name of the model to use (optional)", default=DEFAULT_MODEL)
    run_parser.add_argument("-s", "--server", type=str, choices=AVAILABLE_SERVERS, help="name of the server to use (optional)", default=DEFAULT_SERVER)
    run_parser.add_argument("--verbose", action="store_true", help="enable verbose output", default=False)
    
    # render mode
    render_parser = subparsers.add_parser('render', help="Render from prompt or file")
    input_group = render_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-p", "--prompt", type=str, help="input prompt (max 500 characters)")
    input_group.add_argument("-f", "--file", type=str, help="markdown file to edit")

    render_parser.add_argument("-m", "--model_name", type=str, choices=AVAILABLE_MODELS_GROQ, help="name of the model to use (optional)", default=DEFAULT_MODEL)
    render_parser.add_argument("-s", "--server", type=str, choices=AVAILABLE_SERVERS, help="name of the server to use (optional)", default=DEFAULT_SERVER)
    render_parser.add_argument("-sd", "--start_delimiter", type=str, help="start delimiter for the text to replace (optional)", default='{"')
    render_parser.add_argument("-ed", "--end_delimiter", type=str, help="end delimiter for the text to replace (optional)", default='"}')
    render_parser.add_argument("-v", "--version", action="version", version="smoltex v0.2.0")
    render_parser.add_argument("-l", "--list_models", action="store_true", help="list available models")
    render_parser.add_argument("--verbose", action="store_true", help="enable verbose output", default=False)

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # if args.list_models:
    #     list_models()
    #     return

    model_name = args.model_name
    server_type = args.server
    if server_type == "llama-cpp" and not LOCAL_MODEL_AVAILABLE:
        print(f"Warning: server {server_type} not found, using default...")
        server_type = 'groq'
    

    verbose = args.verbose

    llm = create_llm_client(server_type=server_type, model=model_name, verbose=verbose)
    
    if args.mode == 'run':
        listen_clipboard(llm, model_name=model_name)
        
    elif args.mode == 'render':
        start_delimiter = args.start_delimiter
        end_delimiter = args.end_delimiter
        
        if args.file:
            start = time.time()
            process_markdown_latex(args.file, llm, get_latex_response, model_name=model_name, 
                                start_delimiter=start_delimiter, end_delimiter=end_delimiter, verbose=verbose)
            end = time.time()
            verbose_print(f"\n>> completion time: [bold yellow]{(end - start)*1000} ms[/]", verbose)
            
        elif args.prompt:
            prompt = args.prompt.strip()
            if len(prompt) > MAX_PROMPT_LENGTH:
                rich.print(">> [bold red]ArgumentError: prompt must not exceed 500 characters.[/]")
                exit(1)
            start = time.time()
            output = get_latex_response(llm, prompt, model_name=model_name, verbose=verbose)
            end = time.time()
            rich.print(f"\n[bold green]Latex string:[/] ", end="")
            print(output)
            rich.print(f"\n>> completion time: [bold yellow]{(end - start)*1000} ms[/]")
            # render
            try:
                show_latex_equation(output)
            except:
                print("\n>> cannot render equation, maybe try again.")
        
if __name__ == "__main__":
    main()
