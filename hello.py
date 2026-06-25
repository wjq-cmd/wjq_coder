import base64
from cryptography.fernet import Fernet

# 生成密钥（实际使用中应保存密钥）
key = Fernet.generate_key()
cipher = Fernet(key)

# 原始消息
message = "Hello, World!"

# 加密
encrypted = cipher.encrypt(message.encode())
print(f"原始消息: {message}")
print(f"加密密钥: {key.decode()}")
print(f"加密后: {encrypted.decode()}")

# 解密验证（演示用途）
decrypted = cipher.decrypt(encrypted).decode()
print(f"解密后: {decrypted}")