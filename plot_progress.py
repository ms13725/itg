import json
import numpy as np
import matplotlib.pyplot as plt

with open("optimization_history.json", 'r') as f:
    history = json.load(f)

iterations = np.arange(len(history['aspect']))

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(iterations, history['aspect'], color='blue', label='Current $A$')
plt.axhline(6.0, color='red', linestyle='--', label='Target ($A=6$)')
plt.title("Aspect Ratio Progress")
plt.xlabel("Function Evaluations")
plt.ylabel("Aspect Ratio")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(iterations, history['iota'], color='green', label='Current $\iota$')
plt.axhline(0.42, color='red', linestyle='--', label='Target ($\iota=0.42$)')
plt.title("Rotational Transform Progress")
plt.xlabel("Function Evaluations")
plt.ylabel("Mean $\iota$")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(iterations, history['gk_obj'], color='purple', label='Geom Objective Array Norm')
plt.yscale('log')
plt.title("Geometric Objective Progress")
plt.xlabel("Function Evaluations")
plt.ylabel("Norm of Targets")
plt.legend()

plt.tight_layout()
plt.savefig('optimization_progress.pdf')
