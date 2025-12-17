from main8 import text_to_binary, embed_data_pvd, extract_data_pvd, binary_to_text
from PIL import Image
import os


def main():
    base = os.path.dirname(__file__)
    cover_path = os.path.join(base, "cover_test.png")
    stego_path = os.path.join(base, "stego_test.png")

    # Create a small cover image (50x50 RGB)
    img = Image.new('RGB', (50, 50), color=(100, 150, 200))
    img.save(cover_path)

    # Create a large secret intentionally bigger than capacity to test partial extraction handling
    secret = ("ThisIsASecretMessage-" * 100)[:500]

    data_bits = text_to_binary(secret)

    print(f"Secret bytes: {len(secret)}  bits_to_embed: {len(data_bits)}")

    embedded_bits = embed_data_pvd(cover_path, data_bits, stego_path)
    print(f"embed_data_pvd reported embedded bits: {embedded_bits}")

    # Attempt extraction with stop_on_delimiter to avoid reading garbage
    extracted_bits = extract_data_pvd(stego_path, None, stop_on_delimiter=True)
    print(f"extracted bits length: {len(extracted_bits)}")

    decoded = binary_to_text(extracted_bits)
    print(f"decoded length: {len(decoded)}")
    print("decoded preview:\n" + decoded[:500])

    # Compare
    if decoded == secret:
        print("Exact match: full secret recovered")
    else:
        print("Partial or different. First 200 chars:\n" + decoded[:200])


if __name__ == '__main__':
    main()
