import tkinter as tk
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rc

rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def show_latex_equation(latex_string):
    root = tk.Tk()
    root.title("LaTeX Equation")
    root.protocol("WM_DELETE_WINDOW", root.quit)

    fig, ax = plt.subplots(figsize=(4, 2), facecolor='white')
    ax.axis('off')
    
    ax.text(0.5, 0.5, f'${latex_string}$', 
            size=14, ha='center', va='center', 
            transform=ax.transAxes
        )
    
    plt.tight_layout()

    # embed the figure
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    close_button = tk.Button(root, text="Close", command=root.quit)
    close_button.pack(pady=15)

    root.mainloop()

    # Clean up
    plt.close(fig)
    root.destroy()
