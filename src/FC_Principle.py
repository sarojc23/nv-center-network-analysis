import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
m = 1.0        # Mass
omega_g = 1.0  # Frequency of ground state
omega_e = 1.0  # Frequency of excited state
x_shift = 0.7  # Displacement in excited state potential
x = np.linspace(-3, 4.5, 1000)  # Position array

# Harmonic potential energy functions
V_g = 0.5 * m * omega_g**2 * x**2
V_e = 0.5 * m * omega_e**2 * (x - x_shift)**2 + 3  # Shift excited state up for clarity
# Ensure V_g does not exceed 3
V_g[V_g > 3] = None
V_e[V_e > 6] = None
# Function to compute the wavefunction of a harmonic oscillator
def harmonic_wavefunction(n, x, m, omega, hbar=0.5):
    Hn = np.polynomial.hermite.Hermite([0]*n + [1])(np.sqrt(m*omega/hbar) * x)
    psi = np.exp(-m*omega*x**2/(2*hbar)) * Hn
    normalization = np.sqrt(np.sqrt(m*omega/(np.pi*hbar)) / (2**n * np.math.factorial(n)))
    return normalization * psi

# Number of vibrational states to consider
n_max = 3

# Compute wavefunctions and energies
psi_g = [0.2 * harmonic_wavefunction(n, x, m, omega_g) for n in range(n_max + 1)]
E_g = [0.39 * omega_g * (2 * n + 1) for n in range(n_max + 1)]

psi_e = [0.2 * harmonic_wavefunction(n, x - x_shift, m, omega_e) for n in range(n_max + 1)]
E_e = [0.39 * omega_e * (2 * n + 1) + 3 for n in range(n_max + 1)]

# Plot the potentials
plt.figure(figsize=(3, 8))
plt.plot(x, V_g, color='blue',linewidth=1)
plt.plot(x, V_e, color='red',linewidth=1)

# Plot the vibrational levels and wavefunctions
for n in range(n_max + 1):
    plt.plot(x, psi_g[n] + E_g[n], color='blue',linewidth=1)
    plt.plot(x, psi_e[n] + E_e[n], color='red',linewidth=1)
    plt.hlines(E_g[n], -3, 4, colors='blue', linestyles='dotted',linewidth=1)
    plt.hlines(E_e[n], -3, 4, colors='red', linestyles='dotted',linewidth=1)
    # plt.text(-3.5, E_g[n], f'n={n}', fontsize=10, verticalalignment='center')
    # plt.text(4.1, E_g[n], f'n={n}', fontsize=10, verticalalignment='center', horizontalalignment='left')
    plt.text(3.6, E_g[n], f'n={n}', fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    plt.text(3.6, E_e[n], f'm={n}', fontsize=10, verticalalignment='bottom', horizontalalignment='right')




# # Excitation line with arrow
# plt.vlines(-x_shift, E_g[0], E_e[n_max], colors='green', linestyles='dashed')
# # Relaxation line
# plt.vlines(-x_shift, x_shift, E_e[n_max], E_e[0], colors='gray', linestyles='dashed')
# # Recombination line
# plt.vlines(x_shift, E_e[0], E_g[0], colors='red', linestyles='dashed')
# Excitation line with arrow
# Excitation line with arrow
plt.arrow(-x_shift, E_g[0], 0, E_e[n_max] - E_g[0], color='green', linestyle='dashed', linewidth=1,
          head_width=0.1, head_length=0.2, length_includes_head=True)
# plt.annotate('Excitation', xy=(-x_shift, E_e[n_max]), xytext=(-x_shift - 0.5, E_e[n_max] + 0.5),
#              arrowprops=dict(facecolor='black', arrowstyle='->'))

# Relaxation line with arrow from -x_shift to x_shift
plt.arrow(-x_shift, E_e[n_max], 2*x_shift, E_e[0] - E_e[n_max], color='gray', linestyle='dashed', linewidth=1,
          head_width=0.1, head_length=0.1, length_includes_head=True)
# plt.annotate('Relaxation', xy=(0, E_e[n_max]), xytext=(-1, E_e[n_max] + 0.5),
#              arrowprops=dict(facecolor='black', arrowstyle='->'))

# Recombination line from E_e[0] to E_g[0]
plt.arrow(0.5*x_shift, E_e[0], 0, E_g[0] - E_e[0], color='red', linestyle='dashed', linewidth=1,
          head_width=0.1, head_length=0.1, length_includes_head=True)
# Recombination line from E_e[0] to E_g[1]
plt.arrow(x_shift, E_e[0], 0, E_g[1] - E_e[0], color='red', linestyle='dashed', linewidth=1,
          head_width=0.1, head_length=0.1, length_includes_head=True)
# Recombination line from E_e[0] to E_g[1]
plt.arrow(1.5*x_shift, E_e[0], 0, E_g[2] - E_e[0], color='red', linestyle='dashed', linewidth=1,
          head_width=0.1, head_length=0.1, length_includes_head=True)
# plt.annotate('Recombination', xy=(-x_shift, E_e[n_max]), xytext=(-x_shift - 0.5, E_e[n_max] + 0.5),
#              arrowprops=dict(facecolor='black', arrowstyle='->'))

# Add labels and legend with increased font size
plt.xlabel('Position', fontsize=14)
plt.ylabel('Energy', fontsize=14)
# plt.title('Franck-Condon Diagram', fontsize=16)
plt.legend(fontsize=12)
plt.ylim(-0.3, 6.3)
plt.xlim(-2.5, 3.6)

# Remove ticks and tick labels
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

# Save the figure
import os
output_dir = r'C:\Users\Saroj Chand\Documents\GitHub\nv-center-network-analysis\data\FC_daigrams'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "FC.png"), transparent=True, dpi=300)

# Show the plot
plt.grid(False)
plt.show()