#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io

f = "../data/piececonst_r241_N1024_smooth1.mat"

mat = scipy.io.loadmat(f)


# In[6]:


kcoeff = mat["Kcoeff"]
kcoeff_x = mat["Kcoeff_x"]
kcoeff_y = mat["Kcoeff_y"]
sol = mat["sol"]
coeff = mat["coeff"]

print(kcoeff.shape, "kcoeff")
print(kcoeff_x.shape, "kcoeff x")
print(kcoeff_y.shape, "kcoeff y")
print(sol.shape, "sol")
print(coeff.shape, "coeff")


# In[35]:


import matplotlib.pyplot as plt

idx = 0
plt.imshow(kcoeff[idx])
plt.title("kcoeff")
plt.colorbar()
plt.show()

plt.imshow(kcoeff_y[idx])
plt.title("kcoeff_y")
plt.colorbar()
plt.show()

plt.imshow(kcoeff_x[idx])
plt.title("kcoeff_x")
plt.colorbar()
plt.show()

plt.imshow(sol[idx])
plt.title("sol")
plt.colorbar()
plt.show()

plt.imshow(coeff[idx])
plt.title("coeff")
plt.colorbar()
plt.show()


# In[44]:


def darcy_residual(u, a):
    # u: sol, a: piecewise constant diffusion coefficient
    # This is an unbatched implementation
    # (size, size)
    forcing_func = np.ones_like(u)
    D = 1  # 1 uniform grid
    size = u.shape[0]
    dx = D / size
    dy = dx

    ux = np.gradient(u, dx, axis=0)
    uy = np.gradient(u, dy, axis=1)

    aux = a * ux
    auy = a * uy

    auxx = np.gradient(aux, dx, axis=0)
    auyy = np.gradient(auy, dy, axis=1)
    lhs = -(auxx + auyy)
    return lhs - forcing_func


residual = darcy_residual(sol[idx], kcoeff[idx])


# In[45]:


residual.shape


# In[46]:


plt.imshow(residual)
plt.colorbar()


# In[47]:


residual.sum(), residual.mean(), np.linalg.norm(residual)


# In[89]:


# Batched version of darcy residuals
def batched_darcy_residual(u, a):
    # u: sol, a: piecewise constant diffusion coefficient
    # (batch size, size, size)
    forcing_func = np.ones_like(u)
    D = 1  # 1 uniform grid
    bsize = u.shape[0]
    size = u.shape[1]
    dx = D / size
    dy = dx

    ux = np.gradient(u, dx, axis=1)
    uy = np.gradient(u, dy, axis=2)

    aux = a * ux
    auy = a * uy

    auxx = np.gradient(aux, dx, axis=1)
    auyy = np.gradient(auy, dy, axis=2)
    lhs = -(auxx + auyy)
    return lhs - forcing_func


def corrupt_a(a, p: float):
    max_val = a.max().item()
    min_val = a.min().item()
    ax = np.gradient(a, axis=0)
    ay = np.gradient(a, axis=1)
    field = ax + ay

    # Find indices of non-zero entries
    non_zero_indices = np.argwhere(field != 0)

    # Define offsets for neighbors
    offsets = np.array(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    )

    # Calculate all potential neighbor positions for non-zero indices
    neighbor_indices = non_zero_indices[:, None, :] + offsets[None, :, :]

    # Flatten the neighbor_indices array into two columns (row, col)
    neighbor_indices = neighbor_indices.reshape(-1, 2)

    # Filter neighbors to stay within bounds
    valid_mask = (
        (neighbor_indices[:, 0] >= 0)
        & (neighbor_indices[:, 0] < field.shape[0])
        & (neighbor_indices[:, 1] >= 0)
        & (neighbor_indices[:, 1] < field.shape[1])
    )
    valid_neighbors = neighbor_indices[valid_mask]

    # Create a copy of the field to update
    updated_field = a.copy()

    # Set all valid neighbor positions to 1
    rand_vals = np.random.rand(valid_neighbors.shape[0])
    valid_neighbors = valid_neighbors[rand_vals < p]

    # Half of the valid_neighbors are set to max and half are set to min
    valid_neighbors = np.random.permutation(valid_neighbors)
    size = valid_neighbors.shape[0] // 2
    updated_field[valid_neighbors[size:, 0], valid_neighbors[size:, 1]] = max_val
    updated_field[valid_neighbors[:size, 0], valid_neighbors[:size, 1]] = min_val

    return updated_field


# In[79]:


from tqdm import tqdm

corruptions = np.linspace(0, 1, 20, endpoint=True)
print("Corruption vals", corruptions)
residual_corruption_norm = {c: [] for c in corruptions}
residual_corruption_mean = {c: [] for c in corruptions}
for i in tqdm(range(sol.shape[0])):
    u = sol[i]
    a = coeff[i]
    for c in corruptions:
        new_a = corrupt_a(a, p=c)
        residual = darcy_residual(u, new_a)
        residual_corruption_norm[c].append(np.linalg.norm(residual))
        residual_corruption_mean[c].append(np.mean(residual))


# In[81]:


for k in residual_corruption_norm.keys():
    residual_corruption_norm[k] = np.asarray(residual_corruption_norm[k])
    residual_corruption_mean[k] = np.asarray(residual_corruption_mean[k])


# In[86]:


plt.plot(
    residual_corruption_norm.keys(),
    [residual_corruption_norm[k].mean() for k in residual_corruption_norm.keys()],
)
plt.title("Norm based comparison")
plt.xlabel("Percentage of corruption")
plt.ylabel("Mean of residual norm values")
plt.show()
plt.plot(
    residual_corruption_mean.keys(),
    [residual_corruption_mean[k].mean() for k in residual_corruption_mean.keys()],
)
plt.title("Mean based comparison")
plt.xlabel("Percentage of corruption")
plt.ylabel("Mean of Mean residual values")
plt.show()


# In[93]:


plt.imshow(corrupt_a(coeff[idx], p=0.0001))
plt.show()


# In[ ]:
