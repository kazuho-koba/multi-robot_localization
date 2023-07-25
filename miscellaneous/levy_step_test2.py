import numpy as np
import matplotlib.pyplot as plt

gamma = 1.0
alpha = 1.2
min_step = 30
max_step = 450
num_samples = 10000

levy_steps = []
for i in range(num_samples):
    candidate_adopted = False
    max_prob_density = gamma * (min_step ** -alpha)

    while candidate_adopted == False:
        candidate = min_step + (max_step - min_step) * np.random.uniform(0, 1)
        prob_candidate = gamma * (candidate ** -alpha)
        adoption_threshold = np.random.uniform(0, 1) * max_prob_density

        if prob_candidate > adoption_threshold:
            levy_steps.append(candidate)
            candidate_adopted = True


plt.hist(levy_steps, bins=50, edgecolor='black')
plt.title('Levy flight step sizes')
plt.xlabel('Step size')
plt.ylabel('Frequency')
plt.show()