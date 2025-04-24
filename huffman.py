import heapq
from collections import Counter

def build_frequency_table(data):
    """
    Builds a frequency table (histogram) from the input data.
    
    Parameters:
        data (bytes): Input data.

    Returns:
        Counter: Frequency of each byte in the data.
    """
    return Counter(data)

def build_huffman_tree(freq_table):
    """
    Constructs the Huffman tree based on the frequency table.
    
    Parameters:
        freq_table (Counter): Byte frequencies.

    Returns:
        list: Root node of the Huffman tree with symbols and codes.
    """
    heap = [[weight, [symbol, ""]] for symbol, weight in freq_table.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # Edge case: only one symbol in data
        return [1, [heap[0][1][0], "0"]]

    while len(heap) > 1:
        lo, hi = heapq.heappop(heap), heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return heap[0]

def build_huffman_codes(huffman_tree):
    """
    Extracts the Huffman codes from the Huffman tree.
    
    Parameters:
        huffman_tree (list): Huffman tree structure.

    Returns:
        dict: Mapping of symbol to Huffman code (as strings).
    """
    return {symbol: code for symbol, code in huffman_tree[1:]}

def serialize_tree(codes):
    """
    Serializes the Huffman code mapping into bytes for storage.

    Format:
        [num_symbols (2 bytes)] + [symbol][code_length][code_bits...]

    Parameters:
        codes (dict): Huffman codes.

    Returns:
        bytes: Serialized byte representation of Huffman codes.
    """
    result = bytearray(len(codes).to_bytes(2, 'little'))
    for symbol, code in codes.items():
        result.append(symbol)
        result.append(len(code))
        for i in range(0, len(code), 8):
            byte_val = sum((1 << (7-j)) * (code[i+j] == '1') for j in range(8) if i+j < len(code))
            result.append(byte_val)
    return bytes(result)

def deserialize_tree(data):
    """
    Deserializes the Huffman tree from a byte stream.

    Parameters:
        data (bytes): Serialized tree.

    Returns:
        dict: Symbol-to-code mapping.
    """
    codes, pos = {}, 2
    for _ in range(int.from_bytes(data[0:2], 'little')):
        symbol, code_length = data[pos], data[pos+1]
        pos += 2
        code = ''.join('1' if data[pos + i//8] & (1 << (7 - i%8)) else '0' 
                      for i in range(code_length))
        codes[symbol] = code
        pos += (code_length + 7) // 8
    return codes

def efficient_binary_string_to_bytes(binary_string):
    """
    Converts a binary string to a bytes object with padding if needed.

    Parameters:
        binary_string (str): Binary string of 1s and 0s.

    Returns:
        tuple: (bytes object, padding used)
    """
    padding = 8 - (len(binary_string) % 8) if len(binary_string) % 8 else 0
    binary_string += '0' * padding
    return bytes(int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)), padding

def efficient_bytes_to_binary_string(data, padding=0):
    """
    Converts bytes into a binary string and removes padding if specified.

    Parameters:
        data (bytes): Compressed data.
        padding (int): Number of bits to remove from the end.

    Returns:
        str: Binary string.
    """
    result = ''.join(format(byte, '08b') for byte in data)
    return result[:-padding] if padding else result

def compress_huffman(data):
    """
    Compresses the input data using Huffman encoding.

    Parameters:
        data (bytes): Raw input data.

    Returns:
        tuple: (compressed_data, metadata dictionary)
    """
    if len(data) < 100:
        return data, {'compressed': False}

    freq_table = build_frequency_table(data)
    codes = build_huffman_codes(build_huffman_tree(freq_table))
    encoded = ''.join(codes[byte] for byte in data)
    compressed_bytes, padding = efficient_binary_string_to_bytes(encoded)
    tree_bytes = serialize_tree(codes)

    if len(compressed_bytes) + len(tree_bytes) + 10 >= len(data):
        return data, {'compressed': False}

    return compressed_bytes, {
        'huffman_codes': tree_bytes,
        'padding': padding,
        'compressed': True
    }

def decompress_huffman(compressed_data, metadata):
    """
    Decompresses Huffman-compressed data.

    Parameters:
        compressed_data (bytes): Compressed byte stream.
        metadata (dict): Metadata including Huffman codes and padding.

    Returns:
        bytes: Original uncompressed data.
    """
    if not metadata.get('compressed', True):
        return compressed_data

    codes = deserialize_tree(metadata['huffman_codes'])
    lookup = {code: symbol for symbol, code in codes.items()}
    binary_string = efficient_bytes_to_binary_string(compressed_data, metadata.get('padding', 0))

    result, code = bytearray(), ""
    for bit in binary_string:
        code += bit
        if code in lookup:
            result.append(lookup[code])
            code = ""
    return bytes(result)

def compress_image_huffman(flattened_array):
    """
    Compresses a flattened image array using Huffman coding.

    Parameters:
        flattened_array (numpy.ndarray): 1D array of pixel values.

    Returns:
        tuple: (compressed_data, metadata)
    """
    return compress_huffman(flattened_array.tobytes())

def decompress_image_huffman(compressed_data, metadata):
    """
    Decompresses Huffman-compressed image data.

    Parameters:
        compressed_data (bytes): Compressed image data.
        metadata (dict): Huffman metadata.

    Returns:
        bytes: Decompressed image bytes.
    """
    return decompress_huffman(compressed_data, metadata)

def combined_compress(data):
    """
    First compresses data using LZ77, then Huffman for improved efficiency.

    Parameters:
        data (bytes): Raw input data.

    Returns:
        tuple: (compressed_data, combined_metadata)
    """
    from lz77 import compress_lz77
    lz77_compressed, lz77_metadata = compress_lz77(data)
    huffman_compressed, huffman_metadata = compress_huffman(lz77_compressed)
    return huffman_compressed, {
        'lz77': lz77_metadata,
        'huffman': huffman_metadata
    }

def combined_decompress(compressed_data, metadata):
    """
    Decompresses data compressed with LZ77 followed by Huffman coding.

    Parameters:
        compressed_data (bytes): Final compressed output.
        metadata (dict): Contains both Huffman and LZ77 metadata.

    Returns:
        bytes: Decompressed original data.
    """
    from lz77 import decompress_lz77
    huffman_decoded = decompress_huffman(compressed_data, metadata['huffman'])
    return decompress_lz77(huffman_decoded, metadata['lz77'])

def compress_image_combined(flattened_array):
    """
    Compresses an image using combined LZ77 and Huffman coding.

    Parameters:
        flattened_array (numpy.ndarray): Flattened image pixels.

    Returns:
        tuple: (compressed_data, metadata)
    """
    return combined_compress(flattened_array.tobytes())

def decompress_image_combined(compressed_data, metadata):
    """
    Decompresses image data compressed with combined LZ77 + Huffman.

    Parameters:
        compressed_data (bytes): Compressed image data.
        metadata (dict): Metadata containing both LZ77 and Huffman info.

    Returns:
        bytes: Original uncompressed image bytes.
    """
    return combined_decompress(compressed_data, metadata)
