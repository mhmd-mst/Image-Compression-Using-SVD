import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load image and convert it to grayscale and show image
img = plt.imread('image.jpg')
g_weights = np.array([0.3, 0.59, 0.11])
gray_image = img * g_weights
gray_image = gray_image.sum(axis=2)
plt.imshow(gray_image, cmap='gray')
plt.show()

# Apply SVD on the grayscale image
u, s, v = np.linalg.svd(gray_image, full_matrices=False)
print(f'u.shape:{u.shape},s.shape:{s.shape},v.shape:{v.shape}')

# Variance explained top Singular vectors
var_explained = np.round(s ** 2 / np.sum(s ** 2), decimals=6)
print(f'variance Explained by Top 20 singular values:\n{var_explained[0:13]}')
sns.barplot(x=list(range(1, 21)),
            y=var_explained[0:20], color="dodgerblue")
plt.title('Figure 1. Variance Explained Graph')
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.show()

# Choose different number of components and visualize the approximations of the original image
comps = [667, 1, 10, 20, 30, 50]
plt.figure(figsize=(12, 6))
for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]

    if i == 0:
        plt.subplot(2, 3, i + 1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'Actual Image with n_components = {comps[i]}')

    else:
        plt.subplot(2, 3, i + 1),
        plt.imshow(low_rank, cmap='gray'),
        plt.title(f'n_components = {comps[i]}')
plt.show()

# plot relative spectral norm
spectral_norm = [0] * 95
for i in range(5, 100):
    spectral_norm[i - 5] = s[i + 1] / s[0]
plt.title('Figure 2. Relative Spectral Norm')
plt.plot(range(5, 100), spectral_norm )
plt.show()