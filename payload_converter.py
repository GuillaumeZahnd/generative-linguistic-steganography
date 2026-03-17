import torch


def string2bits(payload_string: str) -> list[int]:
    """
    Convert a string payload to a list of bits using UTF-8 encoding.

    Args:
        payload:_string String to encode (e.g., "Hello World").

    Returns:
        List of 0s and 1s representing the payload (8 bits per character).

    Example:
        >>> string2bits("Hi")
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    """
    payload_bytes = payload_string.encode("utf-8")
    bits = []
    for byte in payload_bytes:
        byte_bits = format(byte, "08b")
        bits.extend([int(bit) for bit in byte_bits])
    return bits


def bits2string(payload_bits: list[int]) -> str:
    """
    Convert a list of bits back to a string payload using UTF-8 decoding.

    Args:
        payload_bits: List of 0s and 1s representing the payload.

    Returns:
        Decoded string.

    Raises:
        ValueError: If the number of bits is not a multiple of 8.
        UnicodeDecodeError: If the bits don't form valid UTF-8.

    Example:
        >>> bits2string([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
        'Hi'
    """
    if len(payload_bits) % 8 != 0:
        raise ValueError(
            f"Number of bits ({len(bits)}) must be a multiple of 8. "
            f"Got {len(bits) % 8} remainder bits."
        )

    payload_bytes = []
    for i in range(0, len(payload_bits), 8):
        byte_bits = payload_bits[i:i+8]
        byte_str = ''.join(str(bit) for bit in byte_bits)
        byte_value = int(byte_str, 2)
        payload_bytes.append(byte_value)

    try:
        payload_string = bytes(payload_bytes).decode("utf-8")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"Invalid UTF-8 sequence in decoded bits. {e.reason}"
        )

    return payload_string
