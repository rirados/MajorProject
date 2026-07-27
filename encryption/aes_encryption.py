import os
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
  
    key_path = save_path.replace(".png", "_key.bin")
    with open(key_path, "wb") as f:
        f.write(key)
        f.write(cipher.iv)

    return save_path, key_path
