from typing import Union
import torch
from tqdm import tqdm

from utils import reset_seed
from llm import select_llm
from get_top_probabilities import get_top_probabilities
from cipher import stegano_encoding, stegano_decoding


def generative_encoding(
    payload_bits: list[int],
    prompt: str,
    seed: int,
    model_nickname: str,
    tokens_limit: int,
    top_k: int,
    temperature: float,
    length_header_bits: int,
    dtype: torch.dtype,
    device: Union[str, torch.device]
) -> str:

    reset_seed(seed)

    model, tokenizer = select_llm(model_nickname=model_nickname, dtype=dtype)
    model.eval()

    # Create length header
    length_bits = format(len(payload_bits), f"0{length_header_bits}b")
    length_header = [int(bit) for bit in length_bits]

    # Full message: [length_header] + [payload]
    full_message = length_header + payload_bits

    print(f"Length header: {len(payload_bits)} → {length_header} ({len(length_header)} bits)")
    print(f"Full message: {len(full_message)} bits ({len(length_header)} header + {len(payload_bits)} payload)")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    payload_bit_index = 0

    for token_step in tqdm(range(tokens_limit), desc="Encoding phase"):
        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits[:, -1, :]
        top_probabilities, top_indices = get_top_probabilities(
            logits=logits,
            top_k=top_k,
            temperature=temperature
        )

        # Encoding is only performed while the payload bits have not been entirely consumed
        if payload_bit_index < len(full_message):
            payload_bit = full_message[payload_bit_index]

            selected_token, _ = stegano_encoding(
                top_probabilities=top_probabilities,
                top_indices=top_indices,
                payload_bit=payload_bit
            )

            payload_bit_index += 1

        else:
            next_token_id = torch.argmax(logits, dim=-1)
            selected_token = next_token_id.item()

        next_token_id = torch.tensor([[selected_token]], device=device)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    print(f"\n✓ Finished generation. The payload ({payload_bit_index} bits) was encoded within a text of {len(generated_text)} characters ({token_step + 1} tokens).")

    return generated_text


def generative_decoding(
    generated_text: str,
    prompt: str,
    model_nickname: str,
    top_k: int,
    temperature: float,
    length_header_bits: int,
    seed: int,
    dtype: torch.dtype,
    device: Union[str, torch.device]
) -> list[int]:

    reset_seed(seed)

    model, tokenizer = select_llm(model_nickname=model_nickname, dtype=dtype)
    model.eval()

    # Extract prompt tokens
    prompt_token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    if prompt_token_ids and prompt_token_ids[0] == tokenizer.bos_token_id:
        prompt_token_ids = prompt_token_ids[1:]
    prompt_length = len(prompt_token_ids)

    # Extract generated tokens
    full_token_ids = tokenizer(generated_text, return_tensors="pt").input_ids[0].tolist()
    if full_token_ids and full_token_ids[0] == tokenizer.bos_token_id:
        full_token_ids = full_token_ids[1:]

    # Skip the prompt to isolate the generated tokens
    generated_token_ids = full_token_ids[prompt_length:]

    if not generated_token_ids:
        raise ValueError("No generated tokens found after the prompt.")

    print(f"Decoding from {len(generated_token_ids)} generated tokens...")

    # Start with prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    decoded_bits = []
    skipped_tokens = 0

    for generated_token_id in tqdm(generated_token_ids, desc="Decoding phase"):
        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits[:, -1, :]
        top_probabilities, top_indices = get_top_probabilities(
            logits=logits,
            top_k=top_k,
            temperature=temperature
        )

        # Check if token is in top_k
        top_indices_flat = top_indices.squeeze(0) if top_indices.dim() > 1 else top_indices

        if generated_token_id not in top_indices_flat:
            skipped_tokens += 1
        else:
            try:
                decoded_bit = stegano_decoding(
                    top_probabilities=top_probabilities,
                    top_indices=top_indices,
                    token_vocabulary_index=generated_token_id
                )
                decoded_bits.append(decoded_bit)
            except ValueError:
                skipped_tokens += 1

        # Append token for next iteration
        next_token_id = torch.tensor([[generated_token_id]], device=device)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    # Extract length header
    if len(decoded_bits) < length_header_bits:
        raise ValueError(
            f"Not enough bits decoded ({len(decoded_bits)}) to read length header ({length_header_bits})"
        )

    header_bits = decoded_bits[:length_header_bits]
    header_binary = ''.join(str(bit) for bit in header_bits)
    payload_length = int(header_binary, 2)

    print(f"✓ Length header: {payload_length} bits")

    # Extract only the payload (skip header)
    decoded_payload = decoded_bits[length_header_bits:length_header_bits + payload_length]

    print(f"✓ Decoded {len(decoded_payload)} bits ({skipped_tokens} tokens skipped)")
    print(f"  (Total decoded: {len(decoded_bits)} bits = {length_header_bits} header + {payload_length} payload)")

    return decoded_payload
