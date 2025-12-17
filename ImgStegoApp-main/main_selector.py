import json
import importlib
import tkinter as tk
from tkinter import messagebox

def _safe_import_and_call(module_name: str, func_name: str, *args, **kwargs):
    """Attempt to import module and call func_name; on ImportError show helpful message."""
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        # Suggest installation command
        pkg_hint = "numpy Pillow matplotlib"
        messagebox.showerror(
            "Missing dependency",
            f"Failed to import {module_name}: {e}\n\nInstall required packages and try again:\n\npython -m pip install {pkg_hint}"
        )
        return None

    func = getattr(mod, func_name, None)
    if not func:
        messagebox.showerror("Internal error", f"{module_name}.{func_name} not found")
        return None
    return func(*args, **kwargs)


root = tk.Tk()
root.title("Steganography Algorithm Selection")
root.geometry("400x250")


def show_selector():
    for widget in root.winfo_children():
        widget.destroy()

    tk.Label(
        root,
        text="Select Algorithm",
        font=("Arial", 16, "bold")
    ).pack(pady=30)

    tk.Button(
        root,
        text="LSB Algorithm",
        width=25,
        height=2,
        command=lambda: _safe_import_and_call('main7', 'open_lsb_dashboard', root, show_selector)
    ).pack(pady=10)

    tk.Button(
        root,
        text="PVD Algorithm",
        width=25,
        height=2,
        command=lambda: _safe_import_and_call('main8', 'open_pvd_dashboard', root, show_selector)
    ).pack(pady=10)

    # Exit button to close the application
    tk.Button(
        root,
        text="Exit",
        width=12,
        height=1,
        command=root.destroy
    ).pack(pady=6)


show_selector()
root.mainloop()
