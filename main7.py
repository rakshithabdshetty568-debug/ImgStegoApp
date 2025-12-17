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
# PVD Range Table (Lower Bound, Upper Bound, Bits)
# NOTE: PVD is typically applied only to one channel (Red, index 0, in this implementation)
PVD_RANGES = [
    (0, 7, 3),      # Range width 8, hides 3 bits
    (8, 15, 3),     # Range width 8, hides 3 bits
    (16, 31, 4),    # Range width 16, hides 4 bits
    (32, 63, 5),    # Range width 32, hides 5 bits
    (64, 127, 6),   # Range width 64, hides 6 bits
    (128, 255, 7)   # Range width 128, hides 7 bits
]

# ---------------- Utility functions ----------------
def text_to_binary(text: str) -> str:
    """Converts secret text into a binary string, including a delimiter."""
    delimiter = "\0\0\0"
    binary_delimiter = ''.join(f'{ord(c):08b}' for c in delimiter)
    binary_data = ''.join(f'{ord(c):08b}' for c in text)
    # Payload Type: "00" for Text
    # Add 16-bit length (number of bytes) after payload type to make partial extraction robust
    length_bits = f'{len(text):016b}' if len(text) < (1 << 16) else f'{(len(text) & 0xFFFF):016b}'
    return "00" + length_bits + binary_data + binary_delimiter

def binary_to_text(bits: str) -> str:
    """
    Converts a binary string back to text, robustly stopping at the delimiter.
    Handles non-printable characters for partial extraction by replacing them with '?' 
    or allowing common printable characters.
    """
    delimiter = "\0\0\0"
    delimiter_bits = ''.join(f'{ord(c):08b}' for c in delimiter)

    # 1. Strip the leading payload type header (2 bits)
    if bits.startswith("00"):
        data_bits = bits[2:]
    else:
        return ""

    # Attempt to read a 16-bit length field (if present)
    length_in_bytes = None
    if len(data_bits) >= 16:
        try:
            length_in_bytes = int(data_bits[:16], 2)
            data_bits_after_len = data_bits[16:]
        except Exception:
            length_in_bytes = None
            data_bits_after_len = data_bits
    else:
        data_bits_after_len = data_bits


    # data_bits = bits[2:] if bits.startswith("00") or bits.startswith("01") else bits

    delimiter_found = False

    try:
        # Search for the delimiter
        end_index = data_bits_after_len.index(delimiter_bits)
        data_bits_with_delimiter = data_bits_after_len[:end_index + len(delimiter_bits)]
        delimiter_found = True
    except ValueError:
        # Delimiter not found, message is partial or corrupted. Use available data bits.
        data_bits_with_delimiter = data_bits_after_len

    # 2. Convert to character chunks
    chars = [data_bits_with_delimiter[i:i+8] for i in range(0, len(data_bits_with_delimiter), 8)]

    decoded_text = ''
    if delimiter_found:
        # If the delimiter was found, trust and decode all bytes up to delimiter
        for c in chars:
            if len(c) == 8:
                decoded_text += chr(int(c, 2))
    else:
        # Partial extraction: collect readable characters and skip non-printable bytes
        for c in chars:
            if len(c) != 8:
                continue
            char_code = int(c, 2)
            is_printable_or_control = (32 <= char_code <= 126) or (char_code in [9, 10, 13, 0])
            if is_printable_or_control:
                decoded_text += chr(char_code)
            else:
                # skip non-printable bytes (prevents long runs of '?' in UI)
                continue

    # If the text still contains the delimiter, split it out
    if delimiter in decoded_text:
        decoded_text = decoded_text.split(delimiter)[0]

    # If we had a valid length field, trim or pad the result accordingly
    if length_in_bytes is not None:
        decoded_text = decoded_text[:length_in_bytes]

    return decoded_text

def get_image_bits_required(img_path: str) -> int:
    with Image.open(img_path) as img:
        img = img.convert("RGB") 
        w, h = img.width, img.height
        c = 3
        # Header: Type (2) + W (16) + H (16) + C (8) = 42 bits
        header_bits = 2 + 16 + 16 + 8 
        # Payload: W * H * C * 8 bits
        payload_bits = w * h * c * 8
        return header_bits + payload_bits

def image_to_binary(img_path: str, max_bits=None) -> str:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    w_bin = f'{img.width:016b}'
    h_bin = f'{img.height:016b}'
    c_bin = f'{arr.shape[2]:08b}'
    # Payload Type: "01" for Image
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
    if len(bits) < HEADER_SIZE or not bits.startswith("01"):
        return None
    try:
        if bits[:2] != "01":
        # Not a proper image payload, but try anyway
          pass
        w = int(bits[2:18], 2)
        h = int(bits[18:34], 2)
        c = int(bits[34:42], 2)
    except Exception:
        return None
    if w <= 0 or h <= 0 or c <= 0 or c > 4:
        return None

    data_bits = bits[HEADER_SIZE:]
    num_bytes = len(data_bits) // 8
    if num_bytes == 0:
        return None

    data_bits = data_bits[:num_bytes*8]
    pixels = []
    for i in range(0, len(data_bits), 8):
        byte = int(data_bits[i:i+8], 2)
        pixels.append(byte)

    required_bytes = w * h * c
    if len(pixels) < required_bytes:
        # Pad with zeros if not enough bits (partial extraction)
        pixels.extend([0]*(required_bytes - len(pixels)))
    else:
        pixels = pixels[:required_bytes]

    pixels = np.array(pixels, dtype=np.uint8)
    
    try:
        mode = "RGB" if c==3 else "L" if c==1 else None
        if not mode: return None
        img_arr = pixels.reshape((h, w, c)) if c>1 else pixels.reshape((h,w))
        return Image.fromarray(img_arr, mode)
    except Exception:
        return None


def calculate_mse_psnr(cover_path: str, stego_path: str):
    """
    Calculates MSE, PSNR, altered pixels, and the total pixel values 
    (TPV) for both cover and stego images.
    """
    try:
        cov = Image.open(cover_path).convert("RGB")
        st = Image.open(stego_path).convert("RGB")
    except Exception:
        # Return default values if images cannot be opened
        return 0.0, float('inf'), 0, None, 0, 0
        
    if st.size != cov.size: st = st.resize(cov.size, resample=Image.Resampling.NEAREST)
    
    # Use float arrays for MSE/PSNR calculation
    cov_a_f = np.array(cov, dtype=np.float32)
    st_a_f = np.array(st, dtype=np.float32)
    
    mse = float(np.mean((cov_a_f - st_a_f) ** 2))
    psnr = 10.0 * math.log10((255.0 ** 2) / mse) if mse != 0.0 else float('inf')
    
    # Use uint8 arrays for TPV and altered pixels calculation
    cov_a_uint8 = np.array(cov, dtype=np.uint8)
    st_a_uint8 = np.array(st, dtype=np.uint8)
    
    # Calculate Total Pixel Values (TPV) as the sum of all R+G+B components
    cover_tpv = int(np.sum(cov_a_uint8))
    stego_tpv = int(np.sum(st_a_uint8))
    
    # Check if any channel differs for a pixel
    mask = (cov_a_uint8 != st_a_uint8).any(axis=2)
    altered_pixels = int(mask.sum())
    
    return mse, psnr, altered_pixels, mask, cover_tpv, stego_tpv

# ---------------- PVD Algorithm Functions ----------------

def _get_pvd_range(diff):
    """Finds the range [lk, uk, t] based on the difference 'diff'."""
    for lk, uk, t in PVD_RANGES:
        if lk <= diff <= uk:
            return lk, uk, t
    return 0, 0, 0 

def get_pvd_max_capacity(cover_image_path: str) -> int:
    """
    Calculates the theoretical maximum PVD capacity (bits) for the R channel 
    based on the image dimensions (assuming all pairs can hide 7 bits).
    """
    try:
        img = Image.open(cover_image_path)
        width, height = img.size
        # The Red channel has (width // 2) * height pairs. Max 7 bits per pair.
        num_pairs = (width // 2) * height
        max_capacity_bits = num_pairs * 7
        return max_capacity_bits
    except Exception:
        return 0

def embed_data_pvd(cover_image_path: str, data_bits: str, out_path: str, root=None) -> int:


    """
    Embeds data using PVD in the Red Channel.
    """
    img = Image.open(cover_image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    
    # Process only the Red Channel (index 0)
    arr_r = arr[:, :, 0]
    height, width = arr_r.shape
    
    bits_to_embed = len(data_bits)
    data_idx = 0
    
    # Create a copy to modify
    stego_arr_r = arr_r.copy()
    
    # Iterate over non-overlapping pixel pairs (p_i, p_{i+1})
    for i in range(height):
        for j in range(0, width - 1, 2):
            if data_idx >= bits_to_embed:
                break 
                
            # Cast to int for safe arithmetic
            p_i = int(arr_r[i, j])
            p_i1 = int(arr_r[i, j+1])
            
            d = abs(p_i1 - p_i)
            
            # 1. Determine Range and Bits
            lk, uk, t = _get_pvd_range(d)
            if t == 0: continue

            # Determine actual bits to embed (limited by remaining data)
            t_actual = min(t, bits_to_embed - data_idx)
            if t_actual == 0:
                break

            # 2. Extract Secret Data
            secret_chunk_bin = data_bits[data_idx:data_idx + t_actual]
            # Pad with 0s if we don't have the full t bits (only happens at the end)
            secret_chunk_dec = int(secret_chunk_bin.ljust(t, '0'), 2)
            
            # 3. Calculate New Difference (d')
            d_prime = lk + secret_chunk_dec
            
            # 4. Calculate required modification (m)
            m = d_prime - d
            
            # 5. Modify Pixels (p'_i, p'_{i+1})
            if p_i1 >= p_i:
                p_i_prime = p_i - math.floor(m / 2)
                p_i1_prime = p_i1 + math.ceil(m / 2)
            else: # p_i1 < p_i
                p_i_prime = p_i + math.ceil(m / 2)
                p_i1_prime = p_i1 - math.floor(m / 2)

            # 6. Boundary Check and Readjustment
            # Clip pixel values to [0, 255]
            p_i_prime = np.clip(p_i_prime, 0, 255)
            p_i1_prime = np.clip(p_i1_prime, 0, 255)
            
            # Convert back to NumPy uint8 for assignment
            stego_arr_r[i, j] = np.uint8(p_i_prime)
            stego_arr_r[i, j+1] = np.uint8(p_i1_prime)
            # if data_idx % 5000 == 0:
            #     root.update_idletasks()

            
            data_idx += t_actual
            if root and data_idx % 5000 == 0:
                root.update_idletasks()

        if data_idx >= bits_to_embed:
            break

    # Reassemble the stego image
    stego_arr = arr.copy()
    stego_arr[:, :, 0] = stego_arr_r # Only the Red channel is modified
    
    stego_img = Image.fromarray(stego_arr, "RGB")
    stego_img.save(out_path) 
    
    return data_idx # Returns the total number of bits successfully embedded


def extract_data_pvd(stego_image_path: str, total_bits: Optional[int] = None, stop_on_delimiter: bool = False) -> str:
    """
    Extracts hidden bits from the red channel using PVD.

    - If `total_bits` is provided, extraction stops after that many bits.
    - If `total_bits` is None and `stop_on_delimiter` is True, extraction will stop early when
      a text delimiter ("\0\0\0") is found for text payloads or when an image header + payload
      is fully collected for image payloads.
    """
    img = Image.open(stego_image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    arr_r = arr[:, :, 0]  # RED channel
    h, w = arr_r.shape

    extracted_bits = ""
    bit_count = 0

    # Precompute delimiter bits for text payloads
    delimiter = "\0\0\0"
    delimiter_bits = ''.join(f'{ord(c):08b}' for c in delimiter)

    # Determine maximum bits to extract
    max_bits = total_bits if total_bits is not None else get_pvd_max_capacity(stego_image_path)

    for i in range(h):
        for j in range(0, w - 1, 2):
            if bit_count >= max_bits:
                break

            p_i = int(arr_r[i, j])
            p_i1 = int(arr_r[i, j + 1])

            d = abs(p_i1 - p_i)
            lk, uk, t = _get_pvd_range(d)
            if t == 0:
                continue

            value = d - lk
            bits = format(value, f'0{t}b')

            extracted_bits += bits
            bit_count += t

            # If we're allowed to stop early, check for delimiter or complete image header
            if stop_on_delimiter:
                # Need at least 2 bits to identify payload type
                if len(extracted_bits) >= 2:
                    payload_type_bits = extracted_bits[:2]
                    # Text payload: stop when delimiter is found after payload header
                    if payload_type_bits == "00":
                        # check if delimiter exists in the stream after the 2-bit header
                        if delimiter_bits in extracted_bits[2:]:
                            return extracted_bits[:bit_count]
                    # Image payload: check header (42 bits) and compute expected size
                    elif payload_type_bits == "01" and len(extracted_bits) >= 42:
                        try:
                            w_img = int(extracted_bits[2:18], 2)
                            h_img = int(extracted_bits[18:34], 2)
                            c_img = int(extracted_bits[34:42], 2)
                            required_bits = 42 + (w_img * h_img * c_img * 8)
                            if bit_count >= required_bits:
                                return extracted_bits[:required_bits]
                        except Exception:
                            # If header parsing fails, continue extracting until max_bits
                            pass

        if bit_count >= max_bits:
            break

    return extracted_bits[:max_bits]

# ---------------- Application (GUI/TKinter) ----------------

class StegApp:
    def __init__(self, root, back_callback):
        self.root = root
        self.back_callback=back_callback
        self.setup_main_dashboard()
        self.root.title("PVD Steganography")
        self.NUM_PAIRS = NUM_PAIRS
        # secret_data structure: (Type, Name, Content/Path, Expected Bits, Embedded Bits)
        self.secret_data = [("None", "", "", 0, 0) for _ in range(self.NUM_PAIRS)]
        self.cover_images = [""] * self.NUM_PAIRS
        # Store stego paths persistently (used for auto-selection)
        self._last_stego_paths = [""] * self.NUM_PAIRS 
        self.password = ""
        self.decode_status = [] 
        self.tk_images = [] 
        self.typing_job_id = None # Store the ID for cancellation
        self.setup_main_dashboard()

    # --- Core Tkinter Functions (Helper) ---
    def _insert_char(self, text_widget, content_list, delay):
        """Inserts a single character and schedules the next. (Not used for extracted content)"""
        if content_list:
            char = content_list.pop(0)
            if text_widget.winfo_exists():
                text_widget.insert("end", char)
                self.typing_job_id = self.root.after(delay, lambda: self._insert_char(text_widget, content_list, delay))
            else:
                self.typing_job_id = None
        else:
            if text_widget.winfo_exists():
                text_widget.config(state=tk.DISABLED)
            self.typing_job_id = None

    def _show_typing_effect(self, text_widget, content):
        """
        Schedules the typing effect. 
        (Now bypassed for extracted content to ensure partial text renders correctly)
        """
        if self.typing_job_id:
            try: self.root.after_cancel(self.typing_job_id)
            except Exception: pass
                
        delay = TYPING_DELAY_MS if len(content) <= 1000 else 0 
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", "end")
        content_list = list(content)
        
        self.typing_job_id = self.root.after(10, lambda: self._insert_char(text_widget, content_list, delay))
        return self.typing_job_id 

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
        for w in self.root.winfo_children(): w.destroy()
        tk.Label(self.root, text="PVD Main Dashboard", font=("Arial", 18, "bold")).pack(pady=10)
        tk.Button(self.root, text=f"Encode", width=48, command=self.multi_pair_mode).pack(pady=6)
        tk.Button(self.root, text="Decode", width=48, command=self.decode_section).pack(pady=6)
        tk.Button(self.root,text="Back to Algorithm Selection",font=("Arial", 10, "bold"),command=self.back_callback).pack(side="bottom", pady=10)

    def multi_pair_mode(self):
        for w in self.root.winfo_children(): w.destroy()
        tk.Label(self.root, text=f"Multi Pair Mode (PVD Embeds)", font=("Arial", 16, "bold")).pack(pady=8)

        main_frame = tk.Frame(self.root); main_frame.pack(fill="x", padx=8)
        self.text_entries = []
        self.cover_labels = []
        self.stego_name_entries = []
        
        # Retain secret data on re-entry, but reset embedded bits
        for i in range(self.NUM_PAIRS):
             stype, name, content, exp_bits, _ = self.secret_data[i]
             self.secret_data[i] = (stype, name, content, exp_bits, 0)

        for i in range(self.NUM_PAIRS):
            pair_lf = tk.LabelFrame(main_frame, text=f"Pair #{i+1}", padx=6, pady=6)
            pair_lf.pack(side="left", expand=True, fill="both", padx=6, pady=6)

            cover_f = tk.Frame(pair_lf); cover_f.pack(pady=4)
            tk.Button(cover_f, text="Select Cover", command=lambda idx=i: self.select_cover_image(idx)).pack(side="left")
            
            initial_cover_text = os.path.basename(self.cover_images[i]) if self.cover_images[i] else "No cover selected"
            lbl = tk.Label(cover_f, text=initial_cover_text, wraplength=100)
            self.cover_labels.append(lbl)
            lbl.pack(side="left", padx=4)

            secret_btn_f = tk.Frame(pair_lf); secret_btn_f.pack(pady=4)
            tk.Button(secret_btn_f, text="Select Text", command=lambda idx=i: self.select_secret_input(idx, "Text")).pack(side="left", padx=2)
            tk.Button(secret_btn_f, text="Select Image", command=lambda idx=i: self.select_secret_input(idx, "Image")).pack(side="left", padx=2)

            txt = tk.Text(pair_lf, width=20, height=5)
            self.text_entries.append(txt)
            
            # Restore previous secret data text
            stype = self.secret_data[i][0]
            if stype != "None":
                 name, content_placeholder, required_bits, _ = self.secret_data[i][1:]
                 
                 cover_capacity_msg = ""
                 if self.cover_images[i]:
                    max_cap = get_pvd_max_capacity(self.cover_images[i])
                    cover_capacity_msg = f"Approx. Max PVD Capacity: {max_cap} bits."

                 if stype == "Text":
                    txt.insert("1.0", content_placeholder)
                    txt.config(state=tk.NORMAL)
                 elif stype == "Image":
                    display = f"Secret Image: {name}\nRequired Bits: {required_bits}\n({cover_capacity_msg})"
                    txt.insert("1.0", display)
                    txt.config(state=tk.DISABLED)
            else:
                 txt.insert("1.0", f"Awaiting input (Type: None)")
                 txt.config(state=tk.DISABLED)
                 
            txt.pack(expand=True, fill="both", padx=4, pady=4)

            name_f = tk.Frame(pair_lf); name_f.pack(pady=4)
            tk.Label(name_f, text="Stego Name:").pack(side="left")
            name_entry = tk.Entry(name_f, width=15)
            initial_stego_name = os.path.basename(self._last_stego_paths[i]) if self._last_stego_paths[i] else f"stego_pair_{i+1}_pvd.png"
            name_entry.insert(0, initial_stego_name)
            self.stego_name_entries.append(name_entry)
            name_entry.pack(side="left")

        pwd_f = tk.Frame(self.root); pwd_f.pack(pady=6)
        tk.Label(pwd_f, text="Password:").pack(side="left")
        self.pwd_entry = tk.Entry(pwd_f, show="*", width=30)
        self.pwd_entry.insert(0, self.password)
        self.pwd_entry.pack(side="left", padx=8)

        btnf = tk.Frame(self.root); btnf.pack(pady=8)
        tk.Button(btnf, text=f"Encode", bg="#4caf50", fg="white", command=self.encode_multi_pair).pack(side="left", padx=6)
        tk.Button(btnf, text="Back", command=self.setup_main_dashboard).pack(side="left", padx=6)

    def select_cover_image(self, idx):
        p = filedialog.askopenfilename(
    parent=self.root,
    title=f"Select cover image #{idx+1}",
    filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp"),("All files","*.*")]
)
        self.root.deiconify()
        try:
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
        if not p: return
        
        self.cover_images[idx] = p
        self.cover_labels[idx].config(text=os.path.basename(p))
        
        stype, name, content, required_bits, embedded_bits = self.secret_data[idx]
        if stype in ["Text", "Image"]:
             max_cap = get_pvd_max_capacity(p)
             cover_capacity_msg = f"Approx. Max PVD Capacity: {max_cap} bits."
             
             if stype == "Image":
                 display = f"Secret Image: {name}\nRequired Bits: {required_bits}\n({cover_capacity_msg})"
                 self.text_entries[idx].config(state=tk.NORMAL)
                 self.text_entries[idx].delete("1.0", "end")
                 self.text_entries[idx].insert("1.0", display)
                 self.text_entries[idx].config(state=tk.DISABLED)
             
             # If Text, update required bits if the content was manually typed
             elif stype == "Text":
                 content_text = self.text_entries[idx].get("1.0", "end").rstrip("\n")
                 required_bits = len(text_to_binary(content_text))
                 self.secret_data[idx] = (stype, name, content_text, required_bits, 0)


    def select_secret_input(self, idx, input_type):
        self.text_entries[idx].config(state=tk.NORMAL)
        self.text_entries[idx].delete("1.0", "end")

        if input_type == "Text":
            p = filedialog.askopenfilename(
    parent=self.root,
    title=f"Select secret Text file #{idx+1}",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

            self.root.deiconify()
            try:
                self.root.lift()
                self.root.focus_force()
            except Exception:
                pass

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
                self.text_entries[idx].config(state=tk.NORMAL)
            else:
                manual_text = self.secret_data[idx][2] if self.secret_data[idx][0] == "Text" else ""
                prompt = manual_text if manual_text and manual_text != "Awaiting input (Type: None)" else "Type your secret message here..."
                self.text_entries[idx].insert("1.0", prompt)
                
                self.secret_data[idx] = ("Text", "Manual Input", prompt, len(text_to_binary(prompt)), 0)
                self.text_entries[idx].config(state=tk.NORMAL) 
            
            return

        elif input_type == "Image":
            filetypes = [("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
            self.root.update_idletasks(); self.root.withdraw()
            p = filedialog.askopenfilename(
    parent=self.root,
    title=f"Select secret Image file #{idx+1}",
    filetypes=filetypes
)
            # Restore main window after file dialog
            self.root.deiconify()
            try:
                self.root.lift()
                self.root.focus_force()
            except Exception:
                pass

            if not p:
                self.text_entries[idx].insert("1.0", f"Awaiting input (Type: {self.secret_data[idx][0]})")
                self.text_entries[idx].config(state=tk.DISABLED)
                return

            name = os.path.basename(p)
            try:
                required_bits = get_image_bits_required(p)
                
                cover_capacity_msg = ""
                cover_path = self.cover_images[idx]
                if cover_path:
                    max_cap = get_pvd_max_capacity(cover_path)
                    cover_capacity_msg = f"Approx. Max PVD Capacity: {max_cap} bits."

                display = f"Secret Image: {name}\nRequired Bits: {required_bits}\n({cover_capacity_msg})"

                content = p 
                self.text_entries[idx].insert("1.0", display)
                self.text_entries[idx].config(state=tk.DISABLED)
                self.secret_data[idx] = (input_type, name, content, required_bits, 0)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process secret image: {e}")
                self.text_entries[idx].insert("1.0", "ERROR: Could not process image.")
                self.text_entries[idx].config(state=tk.DISABLED)
                return

    # --- Encoding/Decoding Logic ---
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

            if first_type is None: first_type = stype
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
        try:
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
        if not folder: return

        statuses = []

        for pair in valid_pairs:
            i = pair["index"]
            cover_path = pair["cover_path"]
            total_secret_bits = pair["total_secret_bits"]
            stego_name = self.stego_name_entries[i].get()
            if not stego_name.lower().endswith(('.png', '.bmp')): stego_name += '.png'
            outp = os.path.join(folder, stego_name)
            
            # Get max theoretical capacity
            max_capacity_bits = get_pvd_max_capacity(cover_path)

            try:
                max_capacity = get_pvd_max_capacity(cover_path)
                if pair["type"] == "Text":
                    data_bits = text_to_binary(pair["content"])
                else:  # Image
                    data_bits = image_to_binary(pair["content"], max_bits=max_capacity)

# LIMIT DATA TO CAPACITY (ADD THIS)
                data_bits = data_bits[:max_capacity]
                # if pair["type"] == "Text":
                #     data_bits = text_to_binary(pair["content"])
                # else: # Image
                #     data_bits = image_to_binary(pair["content"], max_bits=total_secret_bits) 

                bits_hidden = embed_data_pvd(cover_path, data_bits, outp)
                # Store the path for automatic selection in decode
                self._last_stego_paths[i] = outp 

                self.secret_data[i] = (pair["type"], pair["name"], pair["content"], total_secret_bits, bits_hidden)

                # --- Capacity/Status Reporting ---
                if bits_hidden >= total_secret_bits and total_secret_bits > 0:
                    statuses.append(f"Secret #{i+1} ({pair['type']}) was **FULLY** hidden. Capacity: {max_capacity_bits} bits | Used: {total_secret_bits} bits.")
                elif bits_hidden < total_secret_bits and bits_hidden > 0:
                    statuses.append(f"Secret #{i+1} ({pair['type']}) was **PARTIALLY** hidden and **SAVED**. Capacity: {max_capacity_bits} bits | Required: {total_secret_bits} bits | Embedded: {bits_hidden} bits.")
                else:
                    statuses.append(f"Secret #{i+1} ({pair['type']}) failed to embed. {total_secret_bits} required, 0 embedded. Capacity: {max_capacity_bits} bits.")

            except Exception as e:
                statuses.append(f"Secret #{i+1} failed embedding into {stego_name}: {e}")

        messagebox.showinfo("Encoding Status", "\n".join(statuses))
        self.decode_section()

    def decode_stego_images(self):
        valid_stego_indices = [i for i, p in enumerate(self._last_stego_paths) if p and os.path.exists(p)]

        if not valid_stego_indices:
            messagebox.showerror("Missing stego", "Select or encode at least one stego image first.")
            return

        self.root.update_idletasks(); self.root.withdraw()
        pwd = simpledialog.askstring("Password", "Enter password:", show="*")
        self.root.deiconify()
        try:
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass

        if pwd is None: return
        if pwd != self.password:
            messagebox.showerror("Wrong password", "You entered wrong password.")
            return

        self.decode_status = []

        for i in valid_stego_indices:
            stego = self._last_stego_paths[i]
            cover_path = self.cover_images[i] 
            
            expected_bits = self.secret_data[i][3] if i < len(self.secret_data) else 0
            embedded_bits = self.secret_data[i][4] if i < len(self.secret_data) else 0

            # Prompt for cover if missing for metrics calculation
            if not cover_path or not os.path.exists(cover_path):
                self.root.update_idletasks(); self.root.withdraw()
                cover_path = filedialog.askopenfilename(title=f"Select original cover for stego #{i+1} ({os.path.basename(stego)})", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
                self.root.deiconify()
                try:
                    self.root.lift()
                    self.root.focus_force()
                except Exception:
                    pass
                
                if not cover_path: 
                    messagebox.showwarning("Skipping", f"Cannot calculate metrics for Stego #{i+1} without original cover. Skipping pair.")
                    continue
                self.cover_images[i] = cover_path 

            st_img = Image.open(stego).convert("RGB")
            # Max capacity for R channel only: 7 bits per 2 pixels
            # max_bits_to_decode = int(st_img.width * st_img.height / 2 * 7)

            # raw_bits = decode_data_pvd(stego, max_bits_to_decode)
            # FIX: decode only embedded bits
            max_capacity = get_pvd_max_capacity(stego)
            embedded_bits = self.secret_data[i][4]
            # If we know how many bits were embedded, extract that many. Otherwise extract
            # incrementally and stop when a payload delimiter or full image payload is found.
            if embedded_bits and embedded_bits > 0:
                raw_bits = extract_data_pvd(stego, embedded_bits)
            else:
                raw_bits = extract_data_pvd(stego, None, stop_on_delimiter=True)

        #    raw_bits = extract_data_pvd(stego, max_bits_to_decode)

            

            PAYLOAD_TYPE_BITS = raw_bits[:2]
            payload_type = "Text" if PAYLOAD_TYPE_BITS == "00" else "Image" if PAYLOAD_TYPE_BITS == "01" else "Unknown"
            
            decoded_content = "N/A"
            extracted_img = None
            actual_decoded_bits = 0
            status_message = "Payload type unknown or invalid."
            extracted_name = self.secret_data[i][1] if i < len(self.secret_data) and self.secret_data[i][1] else os.path.basename(stego)

            if payload_type == "Text":
                decoded_content_raw = binary_to_text(raw_bits)
                
                delimiter = "\0\0\0"
                if delimiter in decoded_content_raw:
                    # Case 1: FULL Extraction (Delimiter Found)
                    decoded_content = decoded_content_raw.split(delimiter)[0]
                    actual_decoded_bits = len(text_to_binary(decoded_content)) 
                    status_message = "Text secret **fully extracted**."
                else:
                    # Case 2: PARTIAL Extraction (Delimiter NOT Found)
                    decoded_content = decoded_content_raw
                    # If delimiter is not found, prefer the actual number of bits we managed to read
                    actual_decoded_bits = len(raw_bits)
                    if embedded_bits and embedded_bits > 0:
                        status_message = f"Text secret **PARTIALLY extracted**. Showing readable data up to embedded length ({embedded_bits} bits)."
                    else:
                        status_message = f"Text secret **PARTIALLY extracted**. Showing readable data available ({actual_decoded_bits} bits)."
                    
                if not extracted_name.lower().endswith(('.txt')): extracted_name += ".txt"

            elif payload_type == "Image":
                # binary_to_image returns the reconstructed PIL image object (even if partial/padded)
                extracted_img = binary_to_image(raw_bits)

                if extracted_img:
                    w = extracted_img.width; h = extracted_img.height; c = 3
                    # Recalculate required payload bits (including header) based on the image size read from header
                    required_payload_bits = w * h * c * 8 + 42 
                    data_bits_available = len(raw_bits)
                    
                    if data_bits_available >= required_payload_bits and w > 0 and h > 0:
                        status_message = f"Image {w}x{h} ({c} Channels) **fully extracted**."
                        actual_decoded_bits = required_payload_bits
                    else:
                        # Partial image extraction - data bits available is the best estimate
                        actual_decoded_bits = data_bits_available
                        # Updated status message for clarity on partial extraction
                        status_message = f"Partial image data extracted: {data_bits_available} bits of required {required_payload_bits} bits. Image shown may be incomplete)."
                else:
                    status_message = "Failed to reconstruct image. Header invalid or missing."
                
                # FIX: Set decoded_content to a non-text marker to suppress the text display window.
                decoded_content = "Image data processed."
                
                if not extracted_name.lower().endswith(('.png', '.bmp', '.jpg')): extracted_name += ".png"
                
            # Recalculate TPV/MSE/PSNR now that cover_path is guaranteed
            mse, psnr, altered_pixels, mask, cover_tpv, stego_tpv = calculate_mse_psnr(cover_path, stego)

            self.decode_status.append({
                "cover_path": cover_path, "stego_path": stego, "secret_type": payload_type,
                "extracted_name": extracted_name, "decoded_content": decoded_content, 
                "extracted_img": extracted_img, "expected_bits": expected_bits,
                "embedded_bits": embedded_bits, "decoded_bits": actual_decoded_bits,
                "mse": mse, "psnr": psnr, "altered_pixels": altered_pixels, "diff_mask": mask,
                "cover_tpv": cover_tpv, "stego_tpv": stego_tpv, "status_message": status_message 
            })

        self.show_report_table_window()

    def decode_section(self):
        for w in self.root.winfo_children(): w.destroy()
        tk.Label(self.root, text="Decode", font=("Arial", 16, "bold")).pack(pady=8)
        stego_frame = tk.Frame(self.root); stego_frame.pack(pady=6)
        tk.Label(stego_frame, text="Select Stego Images for Decoding:").pack()

        self.stego_labels = []
        for i in range(self.NUM_PAIRS):
            s_f = tk.Frame(stego_frame); s_f.pack(pady=2)
            tk.Button(s_f, text=f"Select Stego #{i+1}", width=15, command=lambda idx=i: self.select_stego_image(idx)).pack(side="left")

            # Use _last_stego_paths for persistent display/selection
            label_text = os.path.basename(self._last_stego_paths[i]) if self._last_stego_paths[i] else "No image selected"
            lbl = tk.Label(s_f, text=label_text, width=30, anchor="w")
            self.stego_labels.append(lbl)
            lbl.pack(side="left", padx=4)

        tk.Button(self.root, text="Decode", bg="#2196F3", fg="white", command=self.decode_stego_images).pack(pady=10)
        tk.Button(self.root, text="Back", command=self.setup_main_dashboard).pack(pady=6)

    def select_stego_image(self, idx):
        self.root.update_idletasks()
        self.root.withdraw()
        p = filedialog.askopenfilename(title=f"Select stego image #{idx+1}", filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp"),("All files", "*.*")])
        self.root.deiconify()
        try:
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
        if not p: return
        self._last_stego_paths[idx] = p 
        self.stego_labels[idx].config(text=os.path.basename(p))

    def _get_quality_status(self, psnr):
        """Determines the quality status based on PSNR value."""
        if psnr is None or math.isinf(psnr): return "PERFECT/INF", "green"
        if psnr >= 40: return "HIGH QUALITY", "green"
        elif psnr >= 30: return "ACCEPTABLE", "orange"
        else: return "LOW QUALITY", "red"

    def show_report_table_window(self):
        if not self.decode_status:
            messagebox.showinfo("No data", "No decoded results to show.")
            return

        primary_type = "Text"
        for d in self.decode_status:
            if d["secret_type"] == "Image": primary_type = "Image"; break

        win = tk.Toplevel(self.root)
        win.title(f"PVD Comparison Report (Secret Type: {primary_type})")
        frame = tk.Frame(win); frame.pack(fill="both", expand=True, padx=8, pady=8)

        cols = ["Cover Image Name", "Stego Image Name", "Cover Max Capacity (Bits)",
                f"Secret {primary_type} Required (Bits)", "Bits Embedded", "Decoded Bits", 
                "PSNR (dB)", "MSE", "Cover Total Pixel Value", "Stego Total Pixel Value", 
                "Encoding Status", "Quality Status"]

        tree = ttk.Treeview(frame, columns=cols, show="headings", height=min(10, len(self.decode_status)))
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, anchor="center")

        # Custom widths
        tree.column("Cover Max Capacity (Bits)", width=120)
        tree.column(f"Secret {primary_type} Required (Bits)", width=120)
        tree.column("Bits Embedded", width=90)
        tree.column("PSNR (dB)", width=70)
        tree.column("Cover Total Pixel Value", width=110) 
        tree.column("Stego Total Pixel Value", width=110) 
        tree.column("Quality Status", width=90)


        tree.pack(fill="both", expand=True)

        for d in self.decode_status:
            if d["secret_type"] != primary_type: continue
            
            cover_path = d["cover_path"]
            max_capacity_bits = get_pvd_max_capacity(cover_path)
            
            secret_bits = d["expected_bits"]
            embedded_bits = d["embedded_bits"]
            psnr = d['psnr']
            
            mse_s = f"{d['mse']:.6f}" if d['mse'] is not None else "-"
            psnr_s = f"{psnr:.3f}" if psnr is not None and not math.isinf(psnr) else "INF"
            
            if embedded_bits == 0 and secret_bits > 0: status = "Failed"
            elif secret_bits > 0 and embedded_bits < secret_bits: status = f"Partial ({(embedded_bits / secret_bits) * 100:.2f}%)"
            elif secret_bits > 0: status = f"Full ({100.0:.2f}%)"
            else: status = "N/A"

            quality_status, color = self._get_quality_status(psnr)

            # Insert all values
            row_values = (os.path.basename(d["cover_path"]), os.path.basename(d["stego_path"]), 
                          max_capacity_bits, secret_bits, embedded_bits, 
                          d["decoded_bits"], psnr_s, mse_s, 
                          f"{d['cover_tpv']:,}", f"{d['stego_tpv']:,}", 
                          status, quality_status)
            
            tree.insert("", "end", values=row_values, tags=(color,))
        
        tree.tag_configure('green', foreground='green')
        tree.tag_configure('orange', foreground='orange')
        tree.tag_configure('red', foreground='red')

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
        
        # Function to handle window close and cancel typing effect
        def on_close():
            if hasattr(self, 'typing_job_id') and self.typing_job_id:
                try:
                    self.root.after_cancel(self.typing_job_id)
                except Exception:
                    pass
            win.destroy()
            
        win.protocol("WM_DELETE_WINDOW", on_close)

        for i, d in enumerate(self.decode_status):
            # 1. Load images from paths
            try: cover_img_orig = Image.open(d["cover_path"]).convert("RGB")
            except Exception: cover_img_orig = None
            try: stego_img_orig = Image.open(d["stego_path"]).convert("RGB")
            except Exception: stego_img_orig = None
            
            # 2. Get the extracted image object directly (this is the potentially partial image)
            extracted_img_obj = d.get('extracted_img')
            
            pair_lf = tk.LabelFrame(scrollable_frame, text=f"Pair #{i+1}: Stego - {os.path.basename(d['stego_path'])} | Secret Type: {d['secret_type']}", padx=10, pady=10)
            pair_lf.pack(padx=10, pady=10, fill="x")

            img_comp_frame = tk.Frame(pair_lf); img_comp_frame.pack(pady=5)
            MAX_DISPLAY_SIZE = (200, 200) 
            
            # Use the actual image objects/paths
            image_data = [
                ("Original Cover", cover_img_orig), 
                ("Stego Image", stego_img_orig), 
                (f"Extracted Secret\n({d['secret_type']})", extracted_img_obj)
            ]
            
            for title, img_to_display in image_data:
                col_frame = tk.Frame(img_comp_frame); col_frame.pack(side="left", padx=15, pady=5)
                display_label_text = title
                
                # FIX: Check if the variable holds a valid PIL Image object
                if isinstance(img_to_display, Image.Image):
                    img_display = img_to_display.copy(); img_display.thumbnail(MAX_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(img_display); self.tk_images.append(tk_img) 
                    img_label = tk.Label(col_frame, image=tk_img); img_label.pack()
                    display_label_text = f"{title}\n{img_to_display.width}x{img_to_display.height}"
                else:
                    img_label = tk.Label(col_frame, text="N/A", width=25, height=10, bg="light gray"); img_label.pack(pady=5)
                    
                tk.Label(col_frame, text=display_label_text, font=("Arial", 9, "bold")).pack(pady=2)

            ttk.Separator(pair_lf, orient="horizontal").pack(fill="x", pady=5)
            status_frame = tk.Frame(pair_lf); status_frame.pack(fill="x", pady=5)
            
            status_text = d.get("status_message") or d["decoded_content"]
            color = "green" if d['secret_type'] == "Text" and "fully extracted" in status_text else "red"
            tk.Label(status_frame, text=f"Decoding Status: {status_text}", fg=color, wraplength=700).pack(pady=5)

            # FIX: Only display the text widget if the secret type is *explicitly* Text
            if d['secret_type'] == "Text" and d["decoded_content"] and d["decoded_content"] != "Image data processed.":
                text_widget = tk.Text(status_frame, width=100, wrap="word", tabs="1c") 
                text_widget.bind("<Tab>", self._force_tab_insert) 
                
                vsb = tk.Scrollbar(status_frame, orient="vertical", command=text_widget.yview)
                text_widget.config(yscrollcommand=vsb.set)
                
                display_content = d["decoded_content"] 
                num_lines = max(5, min(20, (len(display_content) // 100) + 2))
                text_widget.config(height=num_lines) 
                
                vsb.pack(side="right", fill="y")
                text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)
                
                # Instant insertion for robustness
                text_widget.config(state=tk.NORMAL)
                text_widget.delete("1.0", "end")
                text_widget.insert("1.0", display_content)
                text_widget.config(state=tk.DISABLED)
                
                save_frame = tk.Frame(status_frame); save_frame.pack(pady=5)
                tk.Button(save_frame, text="Save Extracted Text File", 
                          command=lambda content=d["decoded_content"], name=d['extracted_name']: self._save_extracted_text(content, name)).pack(side="left", padx=5)

            elif d['secret_type'] == "Image" and extracted_img_obj:
                # If it's an image and an image object exists, show the save button
                save_frame = tk.Frame(status_frame); save_frame.pack(pady=5)
                tk.Button(save_frame, text="Save Extracted Image File", 
                          command=lambda img=extracted_img_obj, name=d['extracted_name']: self._save_extracted_image(img, name)).pack(side="left", padx=5)

        tk.Button(win, text="Close Window", command=on_close).pack(pady=10)

    def _save_extracted_text(self, content, suggested_name):
        self.root.update_idletasks(); self.root.withdraw()
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=suggested_name,
                                             filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        self.root.deiconify()
        try:
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
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
        try:
            self.root.lift()
            self.root.focus_force()
        except Exception:
            pass
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

        plt.suptitle("PSNR & MSE Comparison (PVD Steganography)"); plt.tight_layout(); plt.show() 

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

def open_pvd_dashboard(root, back_callback):
    for widget in root.winfo_children():
        widget.destroy()

    # Start the PVD dashboard fullscreen and allow toggling/escape to exit
    try:
        root.attributes("-fullscreen", True)
    except Exception:
        try:
            root.state('zoomed')
        except Exception:
            pass

    def _toggle_fullscreen(event=None):
        try:
            current = bool(root.attributes("-fullscreen"))
            root.attributes("-fullscreen", not current)
        except Exception:
            try:
                root.state('zoomed' if root.state() != 'zoomed' else 'normal')
            except Exception:
                pass

    root.bind('<Escape>', lambda e: root.attributes("-fullscreen", False))
    root.bind('<F11>', _toggle_fullscreen)

    app = StegApp(root, back_callback)   # ← use your PVD GUI class here
    try:
        root.lift(); root.focus_force()
    except Exception:
        pass

    