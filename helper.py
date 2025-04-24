import os
import struct
import numpy as np
from PIL import Image

def read_bitmap(file_path):
    """
    Reads a BMP image file and returns its pixel data as a NumPy array, 
    the BMP header (first 54 bytes), and the image's width and height.

    Parameters:
        file_path (str): Path to the BMP file.

    Returns:
        tuple: (numpy.ndarray of image, header bytes, width, height)
               Returns (None, None, 0, 0) if an error occurs.
    """
    try:
        img = Image.open(file_path)
        with open(file_path, 'rb') as f:
            return np.array(img), f.read(54), *img.size
    except Exception as e:
        print(f"Error reading bitmap: {e}")
        return None, None, 0, 0

def write_bitmap(file_path, img_array):
    """
    Saves a NumPy array as a BMP image file.

    Parameters:
        file_path (str): Output file path.
        img_array (numpy.ndarray): Image data to be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        Image.fromarray(img_array).save(file_path, format="BMP")
        return True
    except Exception as e:
        print(f"Error writing bitmap: {e}")
        return False

def flatten_image(img_array):
    """
    Flattens a multi-dimensional image array into a 1D array.

    Parameters:
        img_array (numpy.ndarray): Original image array.

    Returns:
        tuple: (1D flattened array, original shape)
    """
    return img_array.flatten(), img_array.shape

def unflatten_image(flat_array, original_shape):
    """
    Reshapes a 1D flattened image array back into its original shape.

    Parameters:
        flat_array (numpy.ndarray): Flattened image array.
        original_shape (tuple): Original shape to reshape to.

    Returns:
        numpy.ndarray: Reshaped image array.
    """
    return np.reshape(flat_array, original_shape)

def save_compressed_file(file_path, compressed_data, metadata):
    """
    Saves compressed image data along with metadata to a binary file.

    Parameters:
        file_path (str): Output file path.
        compressed_data (bytes): Compressed image data.
        metadata (dict): Contains width, height, channels, header_data, encoding_metadata.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(file_path, 'wb') as f:
            # Write image dimensions and channels
            f.write(struct.pack('<IIB', metadata['width'], metadata['height'], metadata['channels']))
            # Write header data length and content
            f.write(struct.pack('<I', len(metadata['header_data'])) + metadata['header_data'])
            # Write encoding metadata length and content
            enc_meta = metadata['encoding_metadata'] if isinstance(metadata['encoding_metadata'], bytes) else bytes(metadata['encoding_metadata'])
            f.write(struct.pack('<I', len(enc_meta)) + enc_meta)
            # Write compressed data length and content
            f.write(struct.pack('<I', len(compressed_data)) + compressed_data)
        return True
    except Exception as e:
        print(f"Error saving compressed file: {e}")
        return False

def load_compressed_file(file_path):
    """
    Loads compressed image data and metadata from a binary file.

    Parameters:
        file_path (str): Path to the compressed file.

    Returns:
        tuple: (compressed_data (bytes), metadata (dict))
               Returns (None, None) if an error occurs.
    """
    try:
        with open(file_path, 'rb') as f:
            # Read width and height (4 bytes each), and channels (1 byte)
            width, height = struct.unpack('<II', f.read(8))
            channels = struct.unpack('<B', f.read(1))[0]
            # Read header data
            header_data = f.read(struct.unpack('<I', f.read(4))[0])
            # Read encoding metadata
            encoding_metadata = f.read(struct.unpack('<I', f.read(4))[0])
            # Read compressed data
            compressed_data = f.read(struct.unpack('<I', f.read(4))[0])
            return compressed_data, {
                'width': width, 
                'height': height, 
                'channels': channels,
                'header_data': header_data, 
                'encoding_metadata': encoding_metadata
            }
    except Exception as e:
        print(f"Error loading compressed file: {e}")
        return None, None

def get_file_size(file_path):
    """
    Returns the file size in bytes.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        int: Size of the file in bytes, or 0 if an error occurs.
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        print(f"Error getting file size: {e}")
        return 0

def bytes_to_binary_string(data):
    """
    Converts bytes to a binary string representation.

    Parameters:
        data (bytes): Input byte data.

    Returns:
        str: Binary string (e.g., '01010101...').
    """
    return ''.join(format(byte, '08b') for byte in data)

def binary_string_to_bytes(binary_string):
    """
    Converts a binary string back to bytes, with padding if necessary.

    Parameters:
        binary_string (str): Binary string (must be multiple of 8 bits).

    Returns:
        bytes: Resulting byte data.
    """
    padding = 8 - (len(binary_string) % 8) if len(binary_string) % 8 else 0
    return bytes(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string) + padding, 8))
