"""
GUI for Bitmap Image Compression using LZ77 + Huffman encoding.
Provides functionality to load BMP images, compress them to a custom .bmc format,
and decompress them back to .bmp for viewing.
"""

import os
import time
import pickle
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import helper
import huffman

def create_gui():
    """Main function to create and run the GUI for image compression."""
    root = tk.Tk()
    root.title("Bitmap Image Compression")
    root.geometry("800x600")

    # GUI state variables
    input_path = tk.StringVar()
    output_path = tk.StringVar()
    status_text = tk.StringVar(value="Ready")
    original_size = tk.StringVar(value="0 bytes")
    compressed_size = tk.StringVar(value="0 bytes")
    compression_ratio = tk.StringVar(value="0:1")
    compression_time = tk.StringVar(value="0 seconds")

    # Image preview widgets (initialized later)
    original_image_label = None
    compressed_image_label = None

    # ---------- GUI Layout ---------- #

    # Main frame for the whole UI
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Input/Output path selection frame
    input_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
    input_frame.pack(fill=tk.X, pady=5)

    # Input file widgets
    ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, sticky=tk.W)
    ttk.Entry(input_frame, textvariable=input_path, width=50).grid(row=0, column=1, sticky=tk.EW)
    ttk.Button(input_frame, text="Browse...", command=lambda: browse_input_file()).grid(row=0, column=2, padx=5)

    # Output file widgets
    ttk.Label(input_frame, text="Output File:").grid(row=1, column=0, sticky=tk.W)
    ttk.Entry(input_frame, textvariable=output_path, width=50).grid(row=1, column=1, sticky=tk.EW)
    ttk.Button(input_frame, text="Browse...", command=lambda: browse_output_file()).grid(row=1, column=2, padx=5)
    input_frame.columnconfigure(1, weight=1)

    # Preview image frame
    preview_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
    preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    ttk.Label(preview_frame, text="Original", style="Header.TLabel").grid(row=0, column=0)
    ttk.Label(preview_frame, text="Compressed/Decompressed", style="Header.TLabel").grid(row=0, column=1)

    original_image_label = tk.Label(preview_frame)
    original_image_label.grid(row=1, column=0, padx=10, pady=10)

    compressed_image_label = tk.Label(preview_frame)
    compressed_image_label.grid(row=1, column=1, padx=10, pady=10)

    preview_frame.columnconfigure(0, weight=1)
    preview_frame.columnconfigure(1, weight=1)
    preview_frame.rowconfigure(1, weight=1)

    # Compression statistics
    stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
    stats_frame.pack(fill=tk.X, pady=5)

    ttk.Label(stats_frame, text="Original Size:").grid(row=0, column=0, sticky=tk.W)
    ttk.Label(stats_frame, textvariable=original_size).grid(row=0, column=1, sticky=tk.W)

    ttk.Label(stats_frame, text="Compressed Size:").grid(row=0, column=2, sticky=tk.W)
    ttk.Label(stats_frame, textvariable=compressed_size).grid(row=0, column=3, sticky=tk.W)

    ttk.Label(stats_frame, text="Compression Ratio:").grid(row=1, column=0, sticky=tk.W)
    ttk.Label(stats_frame, textvariable=compression_ratio).grid(row=1, column=1, sticky=tk.W)

    ttk.Label(stats_frame, text="Processing Time:").grid(row=1, column=2, sticky=tk.W)
    ttk.Label(stats_frame, textvariable=compression_time).grid(row=1, column=3, sticky=tk.W)

    # Action buttons
    button_frame = ttk.Frame(main_frame, padding="10")
    button_frame.pack(fill=tk.X, pady=5)

    ttk.Button(button_frame, text="Compress", command=lambda: compress_image()).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Decompress", command=lambda: decompress_image()).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Exit", command=root.destroy).pack(side=tk.RIGHT, padx=5)

    # Status bar
    status_frame = ttk.Frame(main_frame, padding="5")
    status_frame.pack(fill=tk.X, pady=5)
    ttk.Label(status_frame, textvariable=status_text).pack(side=tk.LEFT)

    # ---------- Helper Functions ---------- #

    def browse_input_file():
        """Open file dialog to select input BMP or BMC file."""
        filepath = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Bitmap Images", "*.bmp"), ("Compressed Files", "*.bmc"), ("All Files", "*.*")]
        )
        if filepath:
            input_path.set(filepath)
            if filepath.endswith('.bmc'):
                output_path.set(filepath.replace('.bmc', '_decompressed.bmp'))
            else:
                output_path.set(filepath.replace('.bmp', '_compressed.bmc'))
            update_preview(filepath, is_original=True)

    def browse_output_file():
        """Open save dialog to specify output file path."""
        filepath = filedialog.asksaveasfilename(
            title="Save Output File",
            defaultextension=".bmc",
            filetypes=[("Compressed Files", "*.bmc"), ("Bitmap Images", "*.bmp"), ("All Files", "*.*")]
        )
        if filepath:
            output_path.set(filepath)

    def update_preview(filepath, is_original):
        """Load and display image preview in the GUI."""
        nonlocal original_image_label, compressed_image_label
        try:
            img = Image.open(filepath) if not filepath.endswith('.bmc') else None
            if img:
                img.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(img)
                if is_original:
                    original_image_label.config(image=photo)
                    original_image_label.image = photo
                    original_size.set(f"{os.path.getsize(filepath)} bytes")
                else:
                    compressed_image_label.config(image=photo)
                    compressed_image_label.image = photo
            else:
                # Clear preview if not a viewable image
                if is_original:
                    original_image_label.config(image='')
                    original_image_label.image = None
        except Exception:
            pass  # Ignore preview errors

    def compress_image():
        """Handles compression of selected image using LZ77 + Huffman."""
        in_file = input_path.get()
        out_file = output_path.get()

        if not in_file or not os.path.exists(in_file):
            messagebox.showerror("Error", "Please select a valid input file")
            return

        if not out_file:
            messagebox.showerror("Error", "Please specify an output file")
            return

        try:
            status_text.set("Compressing...")
            root.update()
            start_time = time.time()

            # Read and flatten the image
            img_array, header_data, width, height = helper.read_bitmap(in_file)
            if img_array is None:
                raise Exception("Failed to read the image")

            flattened_array, original_shape = helper.flatten_image(img_array)
            channels = original_shape[2] if len(original_shape) > 2 else 1

            # Compress using combined method
            compressed_data, encoding_metadata = huffman.compress_image_combined(flattened_array)

            metadata = {
                'width': width,
                'height': height,
                'channels': channels,
                'original_shape': original_shape,
                'header_data': header_data,
                'encoding_metadata': pickle.dumps(encoding_metadata)
            }

            if helper.save_compressed_file(out_file, compressed_data, metadata):
                status_text.set("Compression completed")
                orig_size = os.path.getsize(in_file)
                comp_size = os.path.getsize(out_file)
                ratio = orig_size / comp_size if comp_size > 0 else 0
                original_size.set(f"{orig_size} bytes")
                compressed_size.set(f"{comp_size} bytes")
                compression_ratio.set(f"{ratio:.2f}:1")
                compression_time.set(f"{time.time() - start_time:.2f} seconds")
                messagebox.showinfo("Success", "Image compressed successfully")
            else:
                raise Exception("Failed to save compressed file")
        except Exception as e:
            status_text.set("Compression failed")
            messagebox.showerror("Error", f"Compression failed: {str(e)}")

    def decompress_image():
        """Handles decompression of .bmc file and saves as BMP."""
        in_file = input_path.get()
        if not in_file or not os.path.exists(in_file):
            messagebox.showerror("Error", "Please select a valid input file")
            return

        out_file = filedialog.asksaveasfilename(
            title="Save Decompressed Image",
            defaultextension=".bmp",
            filetypes=[("Bitmap Images", "*.bmp")]
        )
        if not out_file:
            return

        try:
            status_text.set("Decompressing...")
            root.update()
            start_time = time.time()

            # Load compressed file and metadata
            compressed_data, metadata = helper.load_compressed_file(in_file)
            if compressed_data is None or metadata is None:
                raise Exception("Failed to read compressed file or metadata")

            encoding_metadata = pickle.loads(metadata['encoding_metadata'])
            decompressed_data = huffman.decompress_image_combined(compressed_data, encoding_metadata)

            width = metadata['width']
            height = metadata['height']
            channels = metadata.get('channels', 3)

            expected_size = width * height * channels
            if len(decompressed_data) != expected_size:
                raise Exception(f"Decompressed size mismatch. Expected {expected_size}, got {len(decompressed_data)}")

            # Convert bytes back to numpy image
            img_array = np.frombuffer(decompressed_data, dtype=np.uint8).reshape(
                (height, width, channels) if channels > 1 else (height, width))

            if helper.write_bitmap(out_file, img_array):
                status_text.set("Decompression completed")
                update_preview(out_file, is_original=False)
                orig_size = os.path.getsize(in_file)
                decomp_size = os.path.getsize(out_file)
                ratio = orig_size / decomp_size if decomp_size > 0 else 0
                original_size.set(f"{orig_size} bytes")
                compressed_size.set(f"{decomp_size} bytes")
                compression_ratio.set(f"{ratio:.2f}:1")
                compression_time.set(f"{time.time() - start_time:.2f} seconds")
                messagebox.showinfo("Success", "Image decompressed successfully")
            else:
                raise Exception("Failed to save decompressed image")

        except Exception as e:
            status_text.set("Decompression failed")
            messagebox.showerror("Error", f"Decompression failed: {str(e)}")

    # Start GUI event loop
    root.mainloop()

if __name__ == "__main__":
    create_gui()
