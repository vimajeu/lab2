import numpy as np

def find_eigen(matrix):
    matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError("The object is not a matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix is not square")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if np.allclose(eigenvalues.imag, 0):
        eigenvalues = np.real(eigenvalues)
    if np.allclose(eigenvectors.imag, 0):
        eigenvectors = np.real(eigenvectors)

    print("eigenvalues:\n", eigenvalues)
    print("eigenvectors:\n", eigenvectors)

    equality = True
    for i in range(len(eigenvalues)):
        if not np.allclose(matrix @ eigenvectors[:, i], eigenvalues[i] * eigenvectors[:, i]):
            equality = False
            break

    if equality:
        print("The equality A⋅v=λ⋅v works for all eigenvectors and eigenvalues")
    else:
        print("The equality A⋅v=λ⋅v doesn't work for all eigenvectors and eigenvalues")

    return eigenvalues, eigenvectors

A = np.array([[2, 0, 0],
              [0, 3, 4],
              [0, 4, 9]])
find_eigen(A)