import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from PIL import Image
import numpy as np

def encrypt_image(image_pil, save_path="outputs/encrypted_image.png"):
    # Convert image to numpy array
    image_array = np.array(image_pil.convert("RGB"), dtype=np.uint8)
    image_bytes = image_array.tobytes()

    # Generate a random 16-byte AES key
    key = get_random_bytes(16)

    # Encrypt using AES CBC mode
    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_bytes = cipher.encrypt(pad(image_bytes, AES.block_size))

    # Reshape back to image dimensions
    os.makedirs("outputs", exist_ok=True)
    enc_array = np.frombuffer(encrypted_bytes[:len(image_bytes)], dtype=np.uint8)
    enc_array = enc_array.reshape(image_array.shape)

    # Save encrypted image
    enc_image = Image.fromarray(enc_array, mode="RGB")
    enc_image.save(save_path)

    # Save the key and IV for decryption later
    key_path = save_path.replace(".png", "_key.bin")
    with open(key_path, "wb") as f:
        f.write(key)
        f.write(cipher.iv)

    return save_path, key_path


def decrypt_image(enc_image_path, key_path, save_path="outputs/decrypted_image.png"):
    # Read the encrypted image
    enc_image = Image.open(enc_image_path)
    enc_array = np.array(enc_image, dtype=np.uint8)
    original_shape = enc_array.shape
    enc_bytes = enc_array.tobytes()

    # Read key and IV
    with open(key_path, "rb") as f:
        key = f.read(16)
        iv = f.read(16)

    # Decrypt using AES CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    decrypted_bytes = unpad(cipher.decrypt(enc_bytes), AES.block_size)

    # Reshape back to image
    dec_array = np.frombuffer(decrypted_bytes[:enc_array.size], dtype=np.uint8)
    dec_array = dec_array.reshape(original_shape)

    # Save decrypted image
    os.makedirs("outputs", exist_ok=True)
    dec_image = Image.fromarray(dec_array, mode="RGB")
    dec_image.save(save_path)

    return save_path
