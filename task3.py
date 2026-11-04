import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)),
                                     np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    inv_diagonalized = eigenvectors @ np.diag(1 / eigenvalues) @ np.linalg.inv(eigenvectors)
    decrypted_vector = inv_diagonalized @ encrypted_vector
    decrypted_vector = np.real(decrypted_vector)
    return "".join(chr(int(round(x))) for x in decrypted_vector)

message = "Hello, World!"
n = len(message)

while True:
    key_matrix = np.random.randint(0, 256, (n, n))
    if abs(np.linalg.det(key_matrix)) > 1e-6 and np.linalg.cond(key_matrix) < 1e6:
        break

encrypted = encrypt_message(message, key_matrix)
decrypted = decrypt_message(encrypted, key_matrix)

print("Original Message: ", message)
print("Encrypted Message:", encrypted)
print("Decrypted Message:", decrypted)
assert decrypted == message