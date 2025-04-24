def find_longest_match(data, current_pos, window_size=4096, lookahead_buffer_size=128):
    """
    Finds the longest match of data starting at current_pos within the sliding window.

    Parameters:
        data (bytes): The input byte sequence.
        current_pos (int): The current position in the data to search from.
        window_size (int): Size of the sliding window to search for matches.
        lookahead_buffer_size (int): Max number of bytes to consider for matching.

    Returns:
        tuple: (offset, length, next_char)
            offset (int): Distance from current position to the match start.
            length (int): Length of the matched substring.
            next_char (int or None): Next character following the match, or None if end reached.
    """
    end_of_buffer = min(current_pos + lookahead_buffer_size, len(data))
    if current_pos >= len(data):
        return (0, 0, None)

    # Determine the next character (or None if at the end of data)
    next_char = None if end_of_buffer == len(data) else data[end_of_buffer]

    if current_pos >= end_of_buffer:
        return (0, 0, next_char)

    best_match_length, best_match_offset = 0, 0

    # Search backwards within the window to find the best match
    for i in range(max(0, current_pos - window_size), current_pos):
        match_length = 0
        # Compare byte-by-byte for as long as the data matches
        while (current_pos + match_length < end_of_buffer and
               data[i + match_length] == data[current_pos + match_length]):
            match_length += 1
            # Don't cross into the future (i + match_length should not >= current_pos)
            if i + match_length >= current_pos:
                break
        # Update best match if a longer one is found
        if match_length > best_match_length:
            best_match_length = match_length
            best_match_offset = current_pos - i

    return (
        best_match_offset, best_match_length,
        data[current_pos + best_match_length] if current_pos + best_match_length < len(data) else None
    )

def variable_length_encode(offset, length):
    """
    Encodes offset and length using variable-length encoding for compression.

    Parameters:
        offset (int): The offset to encode.
        length (int): The length to encode.

    Returns:
        bytes: Encoded offset and length.
    """
    result = bytearray()
    
    # Use 1 byte if value is small, 2 bytes with MSB set otherwise
    if offset < 128:
        result.append(offset)
    else:
        result.extend([(offset >> 8) | 0x80, offset & 0xFF])

    if length < 128:
        result.append(length)
    else:
        result.extend([(length >> 8) | 0x80, length & 0xFF])

    return bytes(result)

def variable_length_decode(data, pos):
    """
    Decodes variable-length encoded offset and length from compressed data.

    Parameters:
        data (bytes): Compressed data stream.
        pos (int): Current position in the stream.

    Returns:
        tuple: (offset, length, new_position)
    """
    # Decode offset
    if data[pos] < 128:
        offset = data[pos]
        pos += 1
    else:
        offset = ((data[pos] & 0x7F) << 8) | data[pos + 1]
        pos += 2

    # Decode length
    if data[pos] < 128:
        length = data[pos]
        pos += 1
    else:
        length = ((data[pos] & 0x7F) << 8) | data[pos + 1]
        pos += 2

    return offset, length, pos

def compress_lz77(data, window_size=4096, lookahead_buffer_size=128):
    """
    Compresses data using the LZ77 algorithm.

    Parameters:
        data (bytes): Input byte data to compress.
        window_size (int): Sliding window size.
        lookahead_buffer_size (int): Lookahead buffer size.

    Returns:
        tuple: (compressed_data (bytes), metadata (dict))
    """
    # Skip compression for very small inputs
    if len(data) < 50:
        return data, {'compressed': False}

    result = bytearray()
    current_pos = 0

    while current_pos < len(data):
        offset, length, next_char = find_longest_match(data, current_pos, window_size, lookahead_buffer_size)
        result.extend(variable_length_encode(offset, length))
        result.append(next_char if next_char is not None else 0xFF)
        current_pos += length + 1  # Move past the match and next char

    return (
        bytes(result),
        {
            'window_size': window_size,
            'lookahead_buffer_size': lookahead_buffer_size,
            'compressed': True
        }
    ) if len(result) < len(data) else (data, {'compressed': False})

def decompress_lz77(compressed_data, metadata):
    """
    Decompresses LZ77 compressed data using associated metadata.

    Parameters:
        compressed_data (bytes): The compressed data.
        metadata (dict): Metadata with 'compressed' flag and buffer sizes.

    Returns:
        bytes: The decompressed original data.
    """
    if not metadata.get('compressed', True):
        return compressed_data

    result = bytearray()
    pos = 0

    while pos < len(compressed_data):
        offset, length, pos = variable_length_decode(compressed_data, pos)
        next_char = compressed_data[pos] if pos < len(compressed_data) else None
        pos += 1

        if offset == 0 and length == 0:
            # No match found, only literal
            if next_char != 0xFF:
                result.append(next_char)
        else:
            # Reconstruct the matched sequence
            for _ in range(length):
                result.append(result[-offset])
            if next_char != 0xFF:
                result.append(next_char)

    return bytes(result)

def compress_image_lz77(flattened_array, window_size=4096, lookahead_buffer_size=128):
    """
    Compresses a flattened image array using LZ77.

    Parameters:
        flattened_array (numpy.ndarray): Flattened pixel array (1D).
        window_size (int): Sliding window size for LZ77.
        lookahead_buffer_size (int): Lookahead buffer size.

    Returns:
        tuple: (compressed_data (bytes), metadata (dict))
    """
    return compress_lz77(flattened_array.tobytes(), window_size, lookahead_buffer_size)

def decompress_image_lz77(compressed_data, metadata):
    """
    Decompresses LZ77-compressed image data.

    Parameters:
        compressed_data (bytes): Compressed image data.
        metadata (dict): Metadata including 'compressed' flag.

    Returns:
        bytes: Decompressed image data.
    """
    return decompress_lz77(compressed_data, metadata)
