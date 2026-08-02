import os
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from PIL import Image
import numpy as np

def encrypt_image(image_pil, save_path="outputs/encrypted_image.png"):
    image_array = np.array(image_pil.convert("RGB"), dtype=np.uint8)
    image_bytes = image_array.tobytes()

    key = get_random_bytes(16)

    cipher = AES.new(key, AES.MODE_CBC)
    encrypted_bytes = cipher.encrypt(pad(image_bytes, AES.block_size))

    os.makedirs("outputs", exist_ok=True)
    enc_array = np.frombuffer(encrypted_bytes[:len(image_bytes)], dtype=np.uint8)
    enc_array = enc_array.reshape(image_array.shape)

    enc_image = Image.fromarray(enc_array, mode="RGB")
    enc_image.save(save_path)

    key_path = save_path.replace(".png", "_key.txt")
    with open(key_path, "w") as f:
        f.write("KEY: " + base64.b64encode(key).decode() + "\n")
        f.write("IV: " + base64.b64encode(cipher.iv).decode() + "\n")

    return save_path, key_path


def decrypt_image(enc_image_path, key_path, save_path="outputs/decrypted_image.png"):
    enc_image = Image.open(enc_image_path)
    enc_array = np.array(enc_image, dtype=np.uint8)
    original_shape = enc_array.shape
    enc_bytes = enc_array.tobytes()

    with open(key_path, "r") as f:
        lines = f.readlines()
        key = base64.b64decode(lines[0].strip().replace("KEY: ", ""))
        iv = base64.b64decode(lines[1].strip().replace("IV: ", ""))

    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    padded_enc = pad(enc_bytes, AES.block_size)
    decrypted_bytes = cipher.decrypt(padded_enc)

    dec_array = np.frombuffer(decrypted_bytes[:enc_array.size], dtype=np.uint8)
    dec_array = dec_array.reshape(original_shape)

    os.makedirs("outputs", exist_ok=True)
    dec_image = Image.fromarray(dec_array, mode="RGB")
    dec_image.save(save_path)

    return save_path
