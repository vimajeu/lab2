import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

image_raw = imread("пес патрон.jpg")
print(image_raw.shape)
plt.figure(figsize=(6, 6))
plt.imshow(image_raw)
plt.title("Original")
plt.show()

image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum / image_sum.max()
print(image_bw.max())
plt.figure(figsize=(6, 6))
plt.imshow(image_bw, cmap='gray')
plt.title("Black and white")
plt.show()

def fit_pca(X):
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = np.cov(Xc, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if np.allclose(eigenvalues.imag, 0):
        eigenvalues = np.real(eigenvalues)
    if np.allclose(eigenvectors.imag, 0):
        eigenvectors = np.real(eigenvectors)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return mean, eigenvalues, eigenvectors

def reconstruct(X, mean, eigenvectors, k):
    W = eigenvectors[:, :k]
    Z = (X - mean) @ W
    return Z @ W.T + mean

mean, eigenvalues, eigenvectors = fit_pca(image_bw)

explained = eigenvalues / eigenvalues.sum()
cumulative = np.cumsum(explained) * 100
k95 = int(np.argmax(cumulative >= 95)) + 1
print("Number of components explaining 95% variance:", k95)

plt.figure(figsize=(9, 4))
plt.plot(cumulative)
plt.axhline(95, color="r", linestyle="--")
plt.axvline(k95, color="k", linestyle="--")
plt.xlabel("Principal components")
plt.ylabel("Cumulative explained variance, %")
plt.title("Cumulative Explained Variance")
plt.show()

rec = reconstruct(image_bw, mean, eigenvectors, k95)
plt.figure(figsize=(7, 6))
plt.imshow(np.clip(rec, 0, 1), cmap="gray")
plt.title(f"Reconstruction, {k95} components")
plt.show()

ks = [5, 15, 25, 75, 100, 170]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, k in zip(axes.ravel(), ks):
    ax.imshow(np.clip(reconstruct(image_bw, mean, eigenvectors, k), 0, 1), cmap="gray")
    ax.set_title(f"Components: {k}")
plt.tight_layout()
plt.show()