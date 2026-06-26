import base64
import binascii

def multi_layer_encrypt(text: str) -> str:
    # 第1层：Base64 编码
    encoded = base64.b64encode(text.encode()).decode()
    # 第2层：Base32 编码
    encoded = base64.b32encode(encoded.encode()).decode()
    # 第3层：字符串反转
    encoded = encoded[::-1]
    # 第4层：凯撒移位（可打印ASCII范围内偏移3）
    encoded = ''.join(chr((ord(c) - 32 + 3) % 95 + 32) for c in encoded)
    
    # 第5层：base85 编码
    encoded = base64.b85encode(encoded.encode()).decode()
    # 第6层：hex 编码
    encoded = binascii.hexlify(encoded.encode()).decode()
    # 第7层：字节 XOR + base64 转可见
    xor_bytes = bytes(b ^ 0x5A for b in encoded.encode())
    encoded = base64.b64encode(xor_bytes).decode()
    # 第8层：二次 base64
    encoded = base64.b64encode(encoded.encode()).decode()
    # 第9层：rot13
    rot13_trans = str.maketrans(
        'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        'NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm'
    )
    encoded = encoded.translate(rot13_trans)
    return encoded

encrypted = multi_layer_encrypt("helloworld")
print(encrypted)