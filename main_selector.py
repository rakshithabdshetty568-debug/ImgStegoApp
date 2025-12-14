import tkinter as tk
from main7 import open_lsb_dashboard
from main8 import open_pvd_dashboard


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
        command=lambda: open_lsb_dashboard(root, show_selector)
    ).pack(pady=10)

    tk.Button(
        root,
        text="PVD Algorithm",
        width=25,
        height=2,
        command=lambda: open_pvd_dashboard(root, show_selector)
    ).pack(pady=10)


show_selector()
root.mainloop()
