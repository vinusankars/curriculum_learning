import numpy as np
import os

best, vanilla, reverse_best = 0, 0, 0
b_count, v_count, r_count = 0, 0, 0

for i in os.listdir():
	if i.startswith('best'):
		b_count += 1
		best += np.load(i)[-1]
	elif i.startswith('vanilla'):
		v_count += 1
		vanilla += np.load(i)[-1]
	elif i.startswith('reverse'):
		r_count += 1
		reverse_best += np.load(i)[-1]

print('DCL+ accuracy: {:.5f}, Vanilla accuracy: {:.5f}, DCL- accuracy: {:.5f}'.format(best[1]/b_count\
	, vanilla[1]/v_count, reverse_best[1]/r_count))

print('DCL+ loss: {:.5f}, Vanilla loss: {:.5f}, DCL- loss: {:.5f}'.format(best[0]/b_count\
	, vanilla[0]/v_count, reverse_best[0]/r_count))