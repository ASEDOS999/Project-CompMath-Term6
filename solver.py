import numpy as np

def GM(f, A, n):
	c = 0
	for i in range(0, n - 1):
		c = A[i + 1, i]/ A[i, i]
		for j in range(min(3, n - i)):
			A[i + 1, i + j] = A[i + 1, i + j] - c * A[i, i + j]
		f[i+1] = (f[i + 1] - f[i] * c)
	x = np.zeros((n,))
	x[n-1] = f[n-1] / A[n-1, n-1]
	for i in range(n-2, -1, -1):
		s = 0
		for j in range(1, min(3, n - i)):
			s += A[i, i + j] * x[i + j]
		x[i] = (f[i] - s) / A[i, i]
	return x

class solver:
	def __init__(self, a, phi, size):
		self.a = a
		self.phi = phi
		self.size = size
	
	def first(self, T, dt, h):
		size = self.size
		phi = self.phi
		N = int(T / dt)
		M = int(size / h)
		L = int(M / 2)
		u = np.array([[phi(i * h, j * h) for i in range(-L, L)] for j in range(-L, L)])
		cur = self.a * dt / h**2
		results = [u.copy()]
		for k in range(N):
			u_1 = u
			for i in range(M):
				a = u[:, i + 1] if i < M - 1 else 0 * u[:, i]
				b = u[:, i - 1] if i > 0 else 0 * u[:, i]
				u_1[:, i] = u[:, i] + cur * (a - 2 * u[:, i] + b)
			u_2 = np.zeros((M, M))
			for i in range(M):
				a = u_1[i + 1, :] if i < M - 1 else 0 * u_1[i, :]
				b = u_1[i - 1, :] if i > 0 else 0 * u_1[i, :]
				u_2[:, i] = u_1[:, i] + cur * (a - 2 * u_1[:, i] + b)
			u = u_2
			results.append(u.copy())
		return results
	
	def second(self, T, dt, h):
		size = self.size
		phi = self.phi
		N = int(T / dt)
		M = int(size / h)
		L = int(M / 2)
		u = np.array([[phi(i * h, j * h) for i in range(-L, L)] for j in range(-L, L)])
		cur = self.a * dt / h**2
		A = np.zeros((M, M))
		a, b, c = cur, -(2 * cur + 1), cur
		for i in range(M):
			A[i, i] = b
			if i < M - 1:
				A[i, i + 1] = c
			if i > 0:
				A[i, i - 1] = a
		results = [u.copy()]
		for k in range(N):
			u_1 = u
			for i in range(M):
				u_1[:, i] = GM(-u.copy()[:, i], A.copy(), M)
			u_2 = np.zeros((M, M))
			for i in range(M):
				u_2[i, :] = GM(-u_1.copy()[i, :], A.copy(), M)
			u = u_2
			results.append(u.copy())
		return results

if __name__ == "__main__":
	res = solver(1, lambda x, y: np.exp(-(x + y)**2), 4).second(1, 0.004, 0.1)
	k = int(len(res) / 10)
	res_ = []
	for i in range(11):
		n = min(i*k, len(res) -1)
		res_.append((n * 0.004, res[n][20, 20], res[n][0, 0]))
	for i in res_:
		print("%.1f %.5f %.5f"%(i[0], i[1], i[2]))
