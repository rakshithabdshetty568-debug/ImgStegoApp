import numpy as np
from PIL import Image, ImageTk 
import os
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from typing import Union, Optional 

# --- Constants ---
NUM_PAIRS = 3
TYPING_DELAY_MS = 1 

# ---------------- Utility functions ----------------
def text_to_binary(text: str) -> str:
    """Converts secret text into a binary string, including a delimiter."""
    delimiter = "\0\0\0"
    binary_delimiter = ''.join(f'{ord(c):08b}' for c in delimiter)
    binary_data = ''.join(f'{ord(c):08b}' for c in text)
    # Add 16-bit length after payload type to make partial extraction robust
    length_bits = f'{len(text):016b}' if len(text) < (1 << 16) else f'{(len(text) & 0xFFFF):016b}'
    return "00" + length_bits + binary_data + binary_delimiter

def binary_to_text(bits: str) -> str:
    """Converts a binary string back to text, stopping at the delimiter."""
    delimiter_bits = ''.join(f'{ord(c):08b}' for c in "\0\0\0")
    if bits.startswith("00"):
        data_bits = bits[2:]
    elif bits.startswith("01"):
        # image payload - not handled here
        return ""
    else:
        return ""

    # Try to read 16-bit length
    length_in_bytes = None
    if len(data_bits) >= 16:
        try:
            length_in_bytes = int(data_bits[:16], 2)
            data_bits_after_len = data_bits[16:]
        except Exception:
            data_bits_after_len = data_bits
    else:
        data_bits_after_len = data_bits

    try:
        end_index = data_bits_after_len.index(delimiter_bits)
        data_bits_with_delimiter = data_bits_after_len[:end_index]
    except ValueError:
        data_bits_with_delimiter = data_bits_after_len

    chars = [data_bits_with_delimiter[i:i+8] for i in range(0, len(data_bits_with_delimiter), 8)]
    decoded = ''.join(chr(int(c, 2)) for c in chars if len(c) == 8 and 32 <= int(c,2) <= 126)
    if length_in_bytes is not None:
        decoded = decoded[:length_in_bytes]
    return decoded

def get_image_bits_required(img_path: str) -> int:
    with Image.open(img_path) as img:
        img = img.convert("RGB") 
        w, h = img.width, img.height
        c = 3
        header_bits = 2 + 16 + 16 + 8 
        payload_bits = w * h * c * 8
        return header_bits + payload_bits

def image_to_binary(img_path: str, max_bits=None) -> str:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    w_bin = f'{img.width:016b}'
    h_bin = f'{img.height:016b}'
    c_bin = f'{arr.shape[2]:08b}'
    header = "01" + w_bin + h_bin + c_bin
    
    if max_bits is None:
        flat_arr = arr.flatten()
        pixel_bits = ''.join(f'{p:08b}' for p in flat_arr)
        return header + pixel_bits
    
    remaining_bits = max_bits - len(header)
    if remaining_bits <= 0: return header 
    max_bytes = remaining_bits // 8
    flat_arr = arr.flatten()[:max_bytes]
    pixel_bits = ''.join(f'{p:08b}' for p in flat_arr)
    return header + pixel_bits

def binary_to_image(bits: str) -> Union[Image.Image, None]: 
    HEADER_SIZE = 42
    if len(bits) < HEADER_SIZE or not bits.startswith("01"): return None
    try:
        w = int(bits[2:18], 2); h = int(bits[18:34], 2); c = int(bits[34:42], 2)
    except Exception: return None
    if w <= 0 or h <= 0 or c <= 0 or c > 4: return None
    required_bytes = w * h * c
    data_bits = bits[HEADER_SIZE:] 
    num_bytes = len(data_bits) // 8
    data_bits = data_bits[:num_bytes * 8]
    if num_bytes == 0: return None

    try:
        byte_list = [int(data_bits[i:i+8], 2) for i in range(0, len(data_bits), 8)]
        pixels = np.array(byte_list, dtype=np.uint8)
        if len(pixels) < required_bytes:
            padding_needed = required_bytes - len(pixels)
            padding = np.zeros(padding_needed, dtype=np.uint8)
            pixels = np.concatenate((pixels, padding))
        pixels = pixels[:required_bytes]
        mode = "RGB" if c == 3 else "L" if c == 1 else None
        if mode:
            image_array = pixels.reshape((h, w, c)) if c > 1 else pixels.reshape((h, w))
            return Image.fromarray(image_array, mode)
    except Exception:
        return None

def embed_data_lsb(cover_image_path: str, data_bits: str, out_path: str) -> int:
    img = Image.open(cover_image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    capacity = arr.size 
    bits_to_embed = len(data_bits)
    if bits_to_embed == 0:
        Image.fromarray(arr, "RGB").save(out_path)
        return 0
    if bits_to_embed > capacity:
        bits_embedded = capacity
        bits_to_use = data_bits[:capacity]
    else:
        bits_embedded = bits_to_embed
        bits_to_use = data_bits
    flat = arr.flatten().copy()
    data_int_array = np.fromiter((1 if ch == '1' else 0 for ch in bits_to_use), dtype=np.uint8, count=len(bits_to_use))
    flat[:bits_embedded] = (flat[:bits_embedded] & 0xFE) | data_int_array 
    stego = flat.reshape(arr.shape)
    Image.fromarray(stego, "RGB").save(out_path)
    return bits_embedded

def decode_data_lsb(stego_path: str, max_bits_to_decode: int) -> np.ndarray:
    """Return LSB bits as a numpy uint8 array of 0/1 values (fast)."""
    img = Image.open(stego_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8).flatten()
    elements_to_decode = min(max_bits_to_decode, len(arr))
    arr = arr[:elements_to_decode]
    lsb_bits = arr & 1
    return lsb_bits.astype(np.uint8)

def calculate_mse_psnr(cover_path: str, stego_path: str):
    cov = Image.open(cover_path).convert("RGB")
    st = Image.open(stego_path).convert("RGB")
    if st.size != cov.size: st = st.resize(cov.size, resample=Image.Resampling.NEAREST)
    cov_a_f = np.array(cov, dtype=np.float32)
    st_a_f = np.array(st, dtype=np.float32)
    mse = float(np.mean((cov_a_f - st_a_f) ** 2))
    psnr = 10.0 * math.log10((255.0 ** 2) / mse) if mse != 0.0 else float('inf')
    cov_a_uint8 = np.array(cov, dtype=np.uint8)
    st_a_uint8 = np.array(st, dtype=np.uint8)
    cover_tpv = int(np.sum(cov_a_uint8))
    stego_tpv = int(np.sum(st_a_uint8))
    mask = (cov_a_uint8 != st_a_uint8).any(axis=2)
    altered_pixels = int(mask.sum())
    return mse, psnr, altered_pixels, mask, cover_tpv, stego_tpv

# ---------------- Application (GUI/TKinter) ----------------
class StegApp:
    def __init__(self, root, back_callback):
        self.root = root
        self.back_callback = back_callback
        command=lambda: open_lsb_dashboard(self.root, self.back_callback)
        self.root.title("LSB Steganography")
        self.NUM_PAIRS = NUM_PAIRS
        self.secret_data = [("None", "", "", 0, 0) for _ in range(self.NUM_PAIRS)]
        self.cover_images = [""] * self.NUM_PAIRS
        self.stego_paths = [""] * self.NUM_PAIRS
        self.password = ""
        self.decode_status = [] 
        self.tk_images = [] 
        self.setup_main_dashboard()  

    # --- Core Tkinter Fixes ---
    def _insert_char(self, text_widget, content_list, delay):
        if content_list:
            char = content_list.pop(0)
            text_widget.insert("end", char)
            self.root.after(delay, lambda: self._insert_char(text_widget, content_list, delay))
        else:
            text_widget.config(state=tk.DISABLED)

    def _show_typing_effect(self, text_widget, content):
        """
        Shows typing effect for short content only.
        For long content (to avoid freezing), insert directly.
        """
        max_animated = 500  # chars threshold for animation
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", "end")
        if len(content) <= max_animated:
            content_list = list(content)
            delay = max(1, TYPING_DELAY_MS)
            self.root.after(10, lambda: self._insert_char(text_widget, content_list, delay))
        else:
            text_widget.insert("1.0", content)
            text_widget.config(state=tk.DISABLED)

    def _force_tab_insert(self, event):
        try:
            current_state = event.widget.cget("state")
            event.widget.config(state=tk.NORMAL)
            event.widget.insert(tk.INSERT, "\t")
            event.widget.config(state=current_state)
            return "break" 
        except:
            return "break"

    def setup_main_dashboard(self):
    # Clear screen ONCE
      for w in self.root.winfo_children():
          w.destroy()
      for w in self.root.winfo_children(): w.destroy()
      tk.Label(self.root, text="Main Dashboard", font=("Arial", 18, "bold")).pack(pady=10)
      tk.Button(self.root, text=f"Encode", width=48, command=self.multi_pair_mode).pack(pady=6)
      tk.Button(self.root, text="Decode", width=48, command=self.decode_section).pack(pady=6)
      tk.Button(self.root,text="Back to Algorithm Selection",font=("Arial", 10, "bold"),command=self.back_callback).pack(side="bottom", pady=10)
    def multi_pair_mode(self):
        for w in self.root.winfo_children(): w.destroy()
        tk.Label(self.root, text=f"Multi Pair Mode (LSB Embeds)", font=("Arial", 16, "bold")).pack(pady=8)

        main_frame = tk.Frame(self.root); main_frame.pack(fill="x", padx=8)
        self.text_entries = []
        self.cover_labels = []
        self.stego_name_entries = []
        self.secret_data = [("None", "", "", 0, 0) for _ in range(self.NUM_PAIRS)]

        for i in range(self.NUM_PAIRS):
            pair_lf = tk.LabelFrame(main_frame, text=f"Pair #{i+1}", padx=6, pady=6)
            pair_lf.pack(side="left", expand=True, fill="both", padx=6, pady=6)

            cover_f = tk.Frame(pair_lf); cover_f.pack(pady=4)
            tk.Button(cover_f, text="Select Cover", command=lambda idx=i: self.select_cover_image(idx)).pack(side="left")
            lbl = tk.Label(cover_f, text="No cover selected", wraplength=100)
            self.cover_labels.append(lbl)
            lbl.pack(side="left", padx=4)

            secret_btn_f = tk.Frame(pair_lf); secret_btn_f.pack(pady=4)
            tk.Button(secret_btn_f, text="Select Text", command=lambda idx=i: self.select_secret_input(idx, "Text")).pack(side="left", padx=2)
            tk.Button(secret_btn_f, text="Select Image", command=lambda idx=i: self.select_secret_input(idx, "Image")).pack(side="left", padx=2)

            txt = tk.Text(pair_lf, width=20, height=5)
            self.text_entries.append(txt)
            txt.insert("1.0", f"Awaiting input (Type: None)")
            txt.config(state=tk.DISABLED)
            txt.pack(expand=True, fill="both", padx=4, pady=4)

            name_f = tk.Frame(pair_lf); name_f.pack(pady=4)
            tk.Label(name_f, text="Stego Name:").pack(side="left")
            name_entry = tk.Entry(name_f, width=15)
            name_entry.insert(0, f"stego_pair_{i+1}.png")
            self.stego_name_entries.append(name_entry)
            name_entry.pack(side="left")

        pwd_f = tk.Frame(self.root); pwd_f.pack(pady=6)
        tk.Label(pwd_f, text="Password:").pack(side="left")
        self.pwd_entry = tk.Entry(pwd_f, show="*", width=30)
        self.pwd_entry.pack(side="left", padx=8)

        btnf = tk.Frame(self.root); btnf.pack(pady=8)
        tk.Button(btnf, text=f"Encode", bg="#4caf50", fg="white", command=self.encode_multi_pair).pack(side="left", padx=6)
        tk.Button(btnf, text="Back", command=self.setup_main_dashboard).pack(side="left", padx=6)

    def select_cover_image(self, idx):
        self.root.update_idletasks()
        self.root.withdraw()
        p = filedialog.askopenfilename(title=f"Select cover image #{idx+1}", 
                                       filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp"),("All files","*.*")])
        self.root.deiconify() 
        if not p: return
        
        self.cover_images[idx] = p
        self.cover_labels[idx].config(text=os.path.basename(p))

    def select_secret_input(self, idx, input_type):
        """Allows selection of text/image file OR direct text typing (if input_type is 'Text')."""
        self.text_entries[idx].config(state=tk.NORMAL)
        self.text_entries[idx].delete("1.0", "end")

        if input_type == "Text":
            self.root.update_idletasks(); self.root.withdraw()
            p = filedialog.askopenfilename(title=f"Select secret Text file #{idx+1} (or cancel to type manually)", 
                                           filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            self.root.deiconify()

            if p:
                name = os.path.basename(p)
                try:
                    with open(p, "r", encoding="utf-8") as f: content = f.read()
                except Exception:
                    try:
                        with open(p, "r", encoding="latin1") as f: content = f.read()
                    except Exception as e:
                        messagebox.showerror("Error", f"Could not read file: {e}")
                        self.text_entries[idx].insert("1.0", "ERROR: Could not read file.")
                        self.text_entries[idx].config(state=tk.DISABLED)
                        return
                
                self.text_entries[idx].insert("1.0", content)
                self.secret_data[idx] = (input_type, name, content, len(text_to_binary(content)), 0)
                self.text_entries[idx].config(state=tk.NORMAL) # Keep enabled for editing
            else:
                manual_text = self.text_entries[idx].get("1.0", "end").strip()
                self.text_entries[idx].delete("1.0", "end")
                prompt = manual_text or "Type your secret message here..."
                self.text_entries[idx].insert("1.0", prompt)
                self.secret_data[idx] = ("Text", "Manual Input", prompt, len(text_to_binary(prompt)), 0)
                self.text_entries[idx].config(state=tk.NORMAL) 
            return

        elif input_type == "Image":
            filetypes = [("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
            self.root.update_idletasks(); self.root.withdraw()
            p = filedialog.askopenfilename(title=f"Select secret Image file #{idx+1}", filetypes=filetypes)
            self.root.deiconify()

            if not p:
                self.text_entries[idx].insert("1.0", f"Awaiting input (Type: {self.secret_data[idx][0]})")
                self.text_entries[idx].config(state=tk.DISABLED)
                return

            name = os.path.basename(p)
            try:
                required_bits = get_image_bits_required(p)
                cover_capacity = 0
                cover_path = self.cover_images[idx]
                if cover_path:
                    cover = Image.open(cover_path).convert("RGB")
                    cover_capacity = cover.width * cover.height * 3 

                if cover_capacity > 0 and required_bits > cover_capacity:
                    display = f"Secret Image: {name}\nRequired Bits: {required_bits}\n⚠️ WARNING: Cover capacity = {cover_capacity} bits — secret won't fully fit."
                else:
                    display = f"Secret Image: {name}\nRequired Bits: {required_bits}"

                content = p 
                self.text_entries[idx].insert("1.0", display)
                self.text_entries[idx].config(state=tk.DISABLED)
                self.secret_data[idx] = (input_type, name, content, required_bits, 0)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process secret image: {e}")
                self.text_entries[idx].insert("1.0", "ERROR: Could not process image.")
                self.text_entries[idx].config(state=tk.DISABLED)
                return

    def encode_multi_pair(self):
        self.password = self.pwd_entry.get() or ""
        valid_pairs = []

        first_type = None
        for i in range(self.NUM_PAIRS):
            stype, name, content_placeholder, _, _ = self.secret_data[i]
            cover_path = self.cover_images[i]

            if not cover_path: continue

            total_secret_bits = 0
            content_to_use = content_placeholder

            if stype == "Text":
                content_text = self.text_entries[i].get("1.0", "end").rstrip("\n")
                if not content_text or content_text == "Type your secret message here...": continue
                
                total_secret_bits = len(text_to_binary(content_text))
                content_to_use = content_text
                
                if name == "Manual Input":
                    display_name = f"Manual: {content_text[:20]}..." if len(content_text) > 20 else content_text
                    self.secret_data[i] = (stype, display_name, content_to_use, total_secret_bits, 0)
                    name = display_name 

            elif stype == "Image":
                if not content_placeholder or not os.path.exists(content_placeholder): continue
                try:
                    total_secret_bits = get_image_bits_required(content_placeholder)
                except Exception: continue
            else:
                continue

            if first_type is None:
                first_type = stype
            elif first_type != stype:
                messagebox.showerror("Secret Type Mismatch", "Warning: all secrets should be of the same type (Text or Image).")
                return

            valid_pairs.append({
                "index": i, "type": stype, "name": name,
                "content": content_to_use, "cover_path": cover_path,
                "total_secret_bits": total_secret_bits
            })

        if not valid_pairs:
            messagebox.showerror("Incomplete Setup", "Please set up at least one complete pair (Cover + Secret).")
            return

        self.root.update_idletasks(); self.root.withdraw()
        folder = filedialog.askdirectory(title="Select folder to save stego images")
        self.root.deiconify()
        if not folder: return

        self.stego_paths = [""] * self.NUM_PAIRS
        statuses = []

        for pair in valid_pairs:
            i = pair["index"]
            cover_path = pair["cover_path"]
            total_secret_bits = pair["total_secret_bits"]
            stego_name = self.stego_name_entries[i].get()
            if not stego_name.lower().endswith(('.png', '.bmp')): stego_name += '.png'
            outp = os.path.join(folder, stego_name)

            try:
                cover_img = Image.open(cover_path).convert("RGB")
                cover_capacity = cover_img.width * cover_img.height * 3 

                data_bits = text_to_binary(pair["content"]) if pair["type"] == "Text" else image_to_binary(pair["content"], max_bits=cover_capacity)

                bits_hidden = embed_data_lsb(cover_path, data_bits, outp)
                self.stego_paths[i] = outp

                self.secret_data[i] = (pair["type"], pair["name"], pair["content"], total_secret_bits, bits_hidden)

                if bits_hidden >= total_secret_bits and total_secret_bits > 0:
                    statuses.append(f"Secret #{i+1} ({pair['type']}) was FULLY hidden. {total_secret_bits} bits used (capacity: {cover_capacity}).")
                elif bits_hidden < total_secret_bits and bits_hidden > 0:
                    statuses.append(f"Secret #{i+1} ({pair['type']}) was PARTIALLY hidden (capacity: {cover_capacity} bits). Only {bits_hidden} out of {total_secret_bits} bits were hidden.")
                else:
                    statuses.append(f"Secret #{i+1} ({pair['type']}) embedded with {bits_hidden} bits.")

            except Exception as e:
                statuses.append(f"Secret #{i+1} failed embedding into {stego_name}: {e}")

        messagebox.showinfo("Encoding Status", "\n".join(statuses))
        self.decode_section()

    def decode_section(self):
        for w in self.root.winfo_children(): w.destroy()
        tk.Label(self.root, text="Decode", font=("Arial", 16, "bold")).pack(pady=8)
        stego_frame = tk.Frame(self.root); stego_frame.pack(pady=6)
        tk.Label(stego_frame, text="Select Stego Images for Decoding:").pack()

        self.stego_labels = []
        for i in range(self.NUM_PAIRS):
            s_f = tk.Frame(stego_frame); s_f.pack(pady=2)
            tk.Button(s_f, text=f"Select Stego #{i+1}", width=15, command=lambda idx=i: self.select_stego_image(idx)).pack(side="left")

            label_text = os.path.basename(self.stego_paths[i]) if self.stego_paths[i] else "No image selected"
            lbl = tk.Label(s_f, text=label_text, width=30, anchor="w")
            self.stego_labels.append(lbl)
            lbl.pack(side="left", padx=4)

        tk.Button(self.root, text="Decode", bg="#2196F3", fg="white", command=self.decode_stego_images).pack(pady=10)
        tk.Button(self.root, text="Back", command=self.setup_main_dashboard).pack(pady=6)

    def select_stego_image(self, idx):
        self.root.update_idletasks()
        self.root.withdraw()
        p = filedialog.askopenfilename(title=f"Select stego image #{idx+1}", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp"),("All files","*.*")])
        self.root.deiconify()
        if not p: return
        self.stego_paths[idx] = p
        self.stego_labels[idx].config(text=os.path.basename(p))

    def decode_stego_images(self):
        valid_stego_indices = [i for i, p in enumerate(self.stego_paths) if p]
        if not valid_stego_indices:
            messagebox.showerror("Missing stego", "Select at least one stego image first.")
            return

        self.root.update_idletasks(); self.root.withdraw()
        pwd = simpledialog.askstring("Password", "Enter password:", show="*")
        self.root.deiconify()

        if pwd is None: return
        if pwd != self.password:
            messagebox.showerror("Wrong password", "You entered wrong password.")
            return

        self.decode_status = []

        for i in valid_stego_indices:
            stego = self.stego_paths[i]
            cover_path = self.cover_images[i]

            if not cover_path:
                try:
                    self.root.update_idletasks(); self.root.withdraw()
                    cover_path = filedialog.askopenfilename(title=f"Select original cover for stego #{i+1}", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
                    self.root.deiconify()
                    if not cover_path: raise Exception("Cover selection cancelled.")
                    self.cover_images[i] = cover_path
                except:
                    messagebox.showerror("Missing cover", f"Cannot calculate metrics for Stego #{i+1} without original cover.")
                    continue

            st_img = Image.open(stego).convert("RGB")
            max_bits_to_decode = st_img.width * st_img.height * 3

            lsb_bits = decode_data_lsb(stego, max_bits_to_decode)  # numpy array of 0/1
            total_bits_available = lsb_bits.size
            whole_byte_bits = (total_bits_available // 8) * 8
            if whole_byte_bits == 0:
                raw_bits = ''.join(str(b) for b in lsb_bits.tolist())
            else:
                trimmed = lsb_bits[:whole_byte_bits]
                packed = np.packbits(trimmed)  # each is a byte (uint8)
                raw_bits = ''.join(f'{b:08b}' for b in packed)
                if whole_byte_bits < total_bits_available:
                    tail = lsb_bits[whole_byte_bits:]
                    raw_bits += ''.join(str(int(x)) for x in tail.tolist())

            PAYLOAD_TYPE_BITS = raw_bits[:2] if len(raw_bits) >= 2 else ""
            payload_type = "Text" if PAYLOAD_TYPE_BITS == "00" else "Image" if PAYLOAD_TYPE_BITS == "01" else "Unknown"
            
            decoded_content = "N/A"
            extracted_name = self.secret_data[i][1] if i < len(self.secret_data) and self.secret_data[i][1] else os.path.basename(stego)
            extracted_img = None
            actual_decoded_bits = 0
            status_message = "Payload type unknown."

            if payload_type == "Text":
                delimiter_bits = ''.join(f'{ord(c):08b}' for c in "\0\0\0")
                idx_delim = raw_bits.find(delimiter_bits, 2)
                if idx_delim != -1:
                    actual_decoded_bits = idx_delim + len(delimiter_bits)
                    decoded_content = binary_to_text(raw_bits[:actual_decoded_bits])
                    status_message = "Text secret fully extracted (Delimiter found)."
                else:
                    actual_decoded_bits = (len(raw_bits) // 8) * 8
                    decoded_content = binary_to_text(raw_bits[:actual_decoded_bits])
                    status_message = "Text secret partially extracted (delimiter NOT found)."
                if not extracted_name.lower().endswith(('.txt')): extracted_name += ".txt"

            elif payload_type == "Image":
                extracted_img = binary_to_image(raw_bits)
                if extracted_img:
                    w = extracted_img.width; h = extracted_img.height; c = 3
                    required_payload_bits = w * h * c * 8 + 42
                    data_bits_available = len(raw_bits)
                    if data_bits_available >= required_payload_bits and w>0 and h>0:
                        status_message = f"Image {w}x{h} ({c} Channels) fully extracted."
                        actual_decoded_bits = required_payload_bits
                    else:
                        status_message = f"Partial image data extracted: {data_bits_available} bits."
                        actual_decoded_bits = data_bits_available
                else:
                    status_message = "Failed to reconstruct image. Header invalid or missing."
                if not extracted_name.lower().endswith(('.png', '.bmp', '.jpg')): extracted_name += ".png"
                decoded_content = status_message
            
            mse, psnr, altered_pixels, mask, cover_tpv, stego_tpv = calculate_mse_psnr(cover_path, stego)

            expected_bits = self.secret_data[i][3] if i < len(self.secret_data) else 0
            embedded_bits = self.secret_data[i][4] if i < len(self.secret_data) else 0

            self.decode_status.append({
                "cover_path": cover_path, "stego_path": stego, "secret_type": payload_type,
                "extracted_name": extracted_name, "decoded_content": decoded_content, 
                "extracted_img": extracted_img, "expected_bits": expected_bits,
                "embedded_bits": embedded_bits, "decoded_bits": actual_decoded_bits,
                "mse": mse, "psnr": psnr, "altered_pixels": altered_pixels, "diff_mask": mask,
                "cover_tpv": cover_tpv, "stego_tpv": stego_tpv, "status_message": status_message 
            })

        self.show_report_table_window()

    def show_report_table_window(self):
        if not self.decode_status:
            messagebox.showinfo("No data", "No decoded results to show.")
            return

        primary_type = "Text"
        for d in self.decode_status:
            if d["secret_type"] == "Image": primary_type = "Image"; break

        win = tk.Toplevel(self.root)
        win.title(f"Comparison Report (Secret Type: {primary_type})")
        frame = tk.Frame(win); frame.pack(fill="both", expand=True, padx=8, pady=8)

        cols = ["Cover Image Name", f"Secret {primary_type} File Name", "Stego Image Name", "Cover Image Total Bits", f"Secret {primary_type} File Total Bits", "Cover Image Total Pixel Value (R+G+B)", "Stego Image Total Pixel Value (R+G+B)", "Decoded Bits", "PSNR", "MSE", "Status"]

        tree = ttk.Treeview(frame, columns=cols, show="headings", height=min(10, len(self.decode_status)))
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="center")

        tree.column("PSNR", width=70)
        tree.pack(fill="both", expand=True)

        for d in self.decode_status:
            if d["secret_type"] != primary_type: continue
            
            cover_img = Image.open(d["cover_path"]).convert("RGB")
            cover_total_bits = cover_img.width * cover_img.height * 3 * 8
            secret_bits = d["expected_bits"]
            embedded_bits = d["embedded_bits"]

            mse_s = f"{d['mse']:.6f}" if d['mse'] is not None else "-"
            psnr_s = f"{d['psnr']:.3f}" if d['psnr'] is not None and not math.isinf(d['psnr']) else "INF"

            if embedded_bits == 0 and secret_bits > 0: status = "Encoding Failed"
            elif secret_bits > 0 and embedded_bits < secret_bits:
                status = f"Partial ({(embedded_bits / secret_bits) * 100:.2f}%)"
            else: status = f"Full ({100.0:.2f}%)"

            row_values = (os.path.basename(d["cover_path"]), d["extracted_name"], os.path.basename(d["stego_path"]),
                          cover_total_bits, secret_bits, d["cover_tpv"], d["stego_tpv"], d["decoded_bits"],
                          psnr_s, mse_s, status)
            tree.insert("", "end", values=row_values)

        btn_frame = tk.Frame(win); btn_frame.pack(pady=8)
        tk.Button(btn_frame, text="Show Extracted Content (Comparison)", width=32, command=self.show_extracted_content_window).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Show Graph (MSE/PSNR)", width=24, command=self.show_graph_from_report).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Show Altered Pixels Map", width=24, command=self.show_altered_from_report).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Show Pixel Pie Chart", width=24, command=self.show_pixel_pie_from_report).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Back to Dashboard", width=20, command=lambda: (win.destroy(), self.setup_main_dashboard())).pack(side="left", padx=6)

    def show_extracted_content_window(self):
        win = tk.Toplevel(self.root)
        win.title("Extracted Secret Content & Image Comparison")
        self.tk_images = [] 
        canvas = tk.Canvas(win)
        scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for i, d in enumerate(self.decode_status):
            try: cover_img_orig = Image.open(d["cover_path"]).convert("RGB")
            except Exception: cover_img_orig = None
            try: stego_img_orig = Image.open(d["stego_path"]).convert("RGB")
            except Exception: stego_img_orig = None
            extracted_img = d.get('extracted_img')
            
            pair_lf = tk.LabelFrame(scrollable_frame, text=f"Pair #{i+1}: Stego - {os.path.basename(d['stego_path'])} | Secret Type: {d['secret_type']}", padx=10, pady=10)
            pair_lf.pack(padx=10, pady=10, fill="x")

            img_comp_frame = tk.Frame(pair_lf); img_comp_frame.pack(pady=5)
            MAX_DISPLAY_SIZE = (200, 200) 
            image_data = [("Original Cover", cover_img_orig), ("Stego Image", stego_img_orig), (f"Extracted Secret\n({d['secret_type']})", extracted_img)]
            
            for title, img_orig in image_data:
                col_frame = tk.Frame(img_comp_frame); col_frame.pack(side="left", padx=15, pady=5)
                display_label_text = title
                
                if img_orig:
                    img_display = img_orig.copy(); img_display.thumbnail(MAX_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(img_display); self.tk_images.append(tk_img) 
                    img_label = tk.Label(col_frame, image=tk_img); img_label.pack()
                    display_label_text = f"{title}\n{img_orig.width}x{img_orig.height}"
                else:
                    img_label = tk.Label(col_frame, text="N/A", width=25, height=10, bg="light gray"); img_label.pack(pady=5)
                    
                tk.Label(col_frame, text=display_label_text, font=("Arial", 9, "bold")).pack(pady=2)

            ttk.Separator(pair_lf, orient="horizontal").pack(fill="x", pady=5)
            status_frame = tk.Frame(pair_lf); status_frame.pack(fill="x", pady=5)
            
            status_text = d.get("status_message") or d["decoded_content"]
            color = "green" if d['secret_type'] == "Text" and "fully extracted" in status_text.lower() else "red"
            tk.Label(status_frame, text=f"Decoding Status: {status_text}", fg=color, wraplength=700).pack(pady=5)

            if d['secret_type'] == "Text" and d["decoded_content"] and d["decoded_content"] != "N/A":
                # Display horizontal, multiline text using full tab width (wrap='word')
                content = d["decoded_content"]
                text_widget = tk.Text(status_frame, width=100, wrap="word", font=("Arial", 11))
                text_widget.bind("<Tab>", self._force_tab_insert)
                # compute a reasonable height (min 6 lines, max 30)
                approx_lines = max(6, min(30, content.count("\n") + 3))
                text_widget.config(height=approx_lines)
                text_widget.pack(fill="x", padx=5, pady=5)
                # typing effect: only for small text to avoid blocking
                self._show_typing_effect(text_widget, content)
                
                save_frame = tk.Frame(status_frame); save_frame.pack(pady=5)
                tk.Button(save_frame, text="Save Extracted Text File", 
                          command=lambda content=d["decoded_content"], name=d['extracted_name']: self._save_extracted_text(content, name)).pack(side="left", padx=5)

            elif d['secret_type'] == "Image" and extracted_img:
                save_frame = tk.Frame(status_frame); save_frame.pack(pady=5)
                tk.Button(save_frame, text="Save Extracted Image File", 
                          command=lambda img=extracted_img, name=d['extracted_name']: self._save_extracted_image(img, name)).pack(side="left", padx=5)

        tk.Button(win, text="Close Window", command=win.destroy).pack(pady=10)
        win.mainloop()

    def _save_extracted_text(self, content, suggested_name):
        self.root.update_idletasks(); self.root.withdraw()
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=suggested_name,
                                             filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        self.root.deiconify()
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f: f.write(content)
                messagebox.showinfo("Success", f"Text content saved to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save text file: {e}")

    def _save_extracted_image(self, img: Image.Image, suggested_name):
        self.root.update_idletasks(); self.root.withdraw()
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=suggested_name,
                                             filetypes=[("PNG image", "*.png"), ("BMP image", "*.bmp"), ("JPEG image", "*.jpg"), ("All files", "*.*")])
        self.root.deiconify()
        if path:
            try:
                img.save(path)
                messagebox.showinfo("Success", f"Image saved to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image file: {e}")

    def show_graph_from_report(self):
        labels = [os.path.basename(d["stego_path"]) for d in self.decode_status]
        mses = [d["mse"] if d["mse"] is not None else 0.0 for d in self.decode_status]
        psnrs = [d["psnr"] if d["psnr"] is not None and not math.isinf(d["psnr"]) else 100.0 for d in self.decode_status] 

        x = np.arange(len(labels)); width = 0.6
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        
        bars1 = ax1.bar(x, psnrs, width); ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10)
        ax1.set_ylim(0, max(100.0, max(psnrs) * 1.2)); ax1.set_title("PSNR (dB) Comparison"); ax1.set_ylabel("PSNR (dB)")
        for i, b in enumerate(bars1):
            val = psnrs[i]; ax1.text(b.get_x() + b.get_width()/2, val + 1.5, f"{val:.2f}", ha='center', va='bottom', fontsize=9)

        max_mse = max(mses) if mses else 1.0
        bars2 = ax2.bar(x, mses, width); ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=10)
        ax2.set_title("MSE Comparison"); ax2.set_ylabel("MSE"); ax2.set_ylim(0, max(1.0, max_mse * 1.2))
        for i, b in enumerate(bars2):
            val = mses[i]; ax2.text(b.get_x() + b.get_width()/2, val + max_mse*0.02, f"{val:.4f}", ha='center', va='bottom', fontsize=9)

        plt.suptitle("PSNR & MSE Comparison (LSB Steganography)"); plt.tight_layout(); plt.show() 

    def show_altered_from_report(self):
        n = len(self.decode_status)
        cols = n
        if n == 0: messagebox.showinfo("No Data", "No comparison data available."); return

        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        if cols == 1: axes = [axes]

        for i, d in enumerate(self.decode_status):
            cov = Image.open(d["cover_path"]).convert("RGB"); cov_a = np.array(cov, dtype=np.uint8)
            mask = d.get("diff_mask"); altered_pixels = d.get("altered_pixels", 0)

            if mask is not None:
                altered = cov_a.copy(); altered[mask] = [255, 0, 0] 
            else:
                altered = cov_a

            axes[i].imshow(altered); axes[i].set_title(f"Altered Pixels: {os.path.basename(d['stego_path'])}\n{altered_pixels} altered"); axes[i].axis('off')

        plt.suptitle("Altered Pixels Highlighted (red)"); plt.tight_layout(); plt.show() 

    def show_pixel_pie_from_report(self):
        n = len(self.decode_status); cols = n
        if n == 0: messagebox.showinfo("No Data", "No comparison data available."); return

        fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
        if cols == 1: axes = [axes]

        legend_handles = None
        for i, d in enumerate(self.decode_status):
            cov = Image.open(d["cover_path"]).convert("RGB"); total_pixels = cov.width * cov.height
            altered_pixels = d.get("altered_pixels", 0); remaining_pixels = max(total_pixels - altered_pixels, 0)

            if total_pixels == 0:
                 axes[i].text(0, 0, "No Pixels", ha='center', va='center', fontsize=12); axes[i].set_title(os.path.basename(d["stego_path"])); continue

            wedges, texts = axes[i].pie([altered_pixels, remaining_pixels], startangle=90, wedgeprops=dict(width=0.4), colors=['#ff9999', '#66b3ff'], labels=[f"{100*altered_pixels/total_pixels:.2f}%", ""], autopct=None)
            
            axes[i].text(0, 0, f"Altered\nPixels\n{altered_pixels}", ha='center', va='center', fontsize=9); axes[i].set_title(os.path.basename(d["stego_path"]))

            if legend_handles is None: legend_handles = wedges

        if legend_handles is not None:
            fig.legend([legend_handles[0], legend_handles[1]], ["Altered Pixels", "Non Altered Pixels"], loc='lower center', ncol=2)

        plt.suptitle("Pixel Alteration Summary: Altered vs. Non Altered (donut charts)"); plt.tight_layout(rect=[0, 0.05, 1, 0.95]); plt.show() 
        

# ---------------- Run ----------------
def open_lsb_dashboard(root, back_callback):
    for widget in root.winfo_children():
        widget.destroy()
    # Ensure the window opens fullscreen (and allow Escape to exit fullscreen)
    try:
        root.attributes("-fullscreen", True)
    except Exception:
        try:
            root.state('zoomed')
        except Exception:
            pass
    root.bind('<Escape>', lambda e: root.attributes("-fullscreen", False))

    app = StegApp(root, back_callback)   # reuse SAME root window
    
