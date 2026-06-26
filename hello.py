import base64
encrypted = "SGVsbG8gV29ybGQ="
print(base64.b64decode(encrypted).decode())