import numpy as np

def colored(text, color):
	if color == "green":
		return f"\033[92m{text}\033[00m"
	if color == "red":
		return f"\033[91m{text}\033[00m"

def dirak_to_qubit_state(dirak_num):
	return np.array([1+0j, 0+0j]) if dirak_num == 0 else np.array([0+0j, 1+0j])

def qubit_state_to_dirak(state):
	return 0 if state == np.array([1+0j, 0+0j]).all() else 1

def bits_table(n):
	return [bin(x)[-n:].replace('b', '0') for x in range(2**n)]

def direct_sum(a, b):
	sum = np.zeros(np.add(a.shape, b.shape))
	sum[:a.shape[0], :a.shape[1]] = a
	sum[a.shape[0]:, a.shape[1]:] = b
	return sum

class QubitRegister:

	def __init__(self, DEBUG, n):
		self.DEBUG = DEBUG
		self.table = bits_table(n)
		self.state = []
		for _ in range(n):
			for complex_num in dirak_to_qubit_state(0):
				self.state.append(complex_num)
		self.n = n
		state = np.array([self.state[0], self.state[1]])
		for index in range(2, (self.n*2)-1, 2):
			t = np.array([
				self.state[index],
				self.state[index+1]
			])
			state = np.kron(state, t)
		self.state = state

	def gate(self, A, qubit_ids):
		k = len(qubit_ids)

		I_GATE = np.array([
			[1, 0],
			[0, 1]
		])

		U = np.zeros((2**self.n, 2**self.n), dtype=np.complex128)

		if k > 1:
			control = qubit_ids[0]
			target = qubit_ids[1]

		def get_indexes(element, A):
			q1 = dirak_to_qubit_state(int(element[control]))
			q2 = dirak_to_qubit_state(int(element[target]))
			state = np.kron(q1, q2)
			state = A @ state
			state = list(state)
			table = bits_table(2)
			basics = []
			for s in state:
				if s != 0+0j:
					basics.append(table[state.index(s)])
					state[state.index(s)] = 0+0j

			elements = []

			# Elements
			for b in basics:
				elements.append(A[int(element[control] + element[target], 2)][int(b, 2)])

			for i in range(len(basics)):
				el = basics[i]
				nel = element
				nel = nel[:control] + el[0] + nel[control+1:]
				nel = nel[:target] + el[1] + nel[target+1:]
				basics[i] = nel

			indexes = []

			# Indexes
			b1 = int(element, 2)
			for b in basics:
				indexes.append([b1, int(b, 2)])

			return [indexes, elements]

		def check(index):
			return True if self.table[index][control] == '1' else False

		def change(i, j):
			return True if check(i) and check(j) else False

		if k == 1:
			target = qubit_ids[0]
			if target == 0:
				U = A
			else:
				U = I_GATE
			for i in range(1, self.n):
				if i == target:
					U = np.kron(U, A)
				else:
					U = np.kron(U, I_GATE)
		else:
			# CHECK
			for i in range(2**self.n):
				for j in range(2**self.n):
					if i == j:
						if change(i, j):
							indexes, elements = get_indexes(self.table[i], A)
							if len(indexes) > 1:
								for i in range(len(elements)):
									i1, j1 = indexes[i]
									i2, j2 = indexes[i]
									el1, el2 = elements
									U[i1][j1] = el1
									U[i2][j2] = el2
							else:
								i2, j2 = indexes[0]
								U[i2][j2] = elements[0]
						else:
							U[i][j] = 1
		self.state = U @ self.state

	def X(self, qubit_id):
		if self.DEBUG:
			print(f"X {qubit_id}")
		X_GATE = np.array([
			[0, 1],
			[1, 0]
		])
		self.gate(X_GATE, [qubit_id])

	def Z(self, qubit_id):
		if self.DEBUG:
			print(f"Z {qubit_id}")
		Z_GATE = np.array([
			[1, 0],
			[0, -1]
		])
		self.gate(Z_GATE, [qubit_id])

	def H(self, qubit_id):
		if self.DEBUG:
			print(f"H {qubit_id}")
		H_GATE = np.array([
			[1, 1],
			[1, -1]
		]) / np.sqrt(2)
		self.gate(H_GATE, [qubit_id])

	def CX(self, controller, target):
		if self.DEBUG:
			print(f"CX {controller} {target}")
		CX_GATE = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 0, 1],
			[0, 0, 1, 0]
		])
		self.gate(CX_GATE, [controller, target])

	def CH(self, controller, target):
		if self.DEBUG:
			print(f"CH {controller} {target}")
		CH_GATE = np.array([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
			[0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
		])
		self.gate(CH_GATE, [controller, target])

	def Ry(self, theta, qubit_id):
		if self.DEBUG:
			print(f"Ry {qubit_id}")
		U = np.array([
			[np.cos(theta/2), -np.sin(theta/2)],
			[np.sin(theta/2), np.cos(theta/2)]
		]) / np.sqrt(2)
		self.gate(U, [qubit_id])

	def measure(self, index):
		if self.DEBUG:
			print(f"MEASURE {index}")

		prob0 = 0
		prob1 = 0

		result = 0

		for t in self.table:
			if t[index] == '1':
				prob1 += abs(self.state[self.table.index(t)]) ** 2
			else:
				prob0 += abs(self.state[self.table.index(t)]) ** 2

		prob = [prob0, prob1]

		prob = np.array(prob)
		prob /= prob.sum()

		if self.DEBUG:
			print(f"PROBABILITIES: {prob}")

		result = np.random.choice([0, 1], p=prob)

		p = [
			np.zeros(2**self.n, dtype=np.complex128),
			np.zeros(2**self.n, dtype=np.complex128)
		]

		for t in self.table:
			if t[index] == '0':
				p[0][self.table.index(t)] = self.state[self.table.index(t)]
			else:
				p[1][self.table.index(t)] = self.state[self.table.index(t)]

		p = np.array(p)

		self.state = p[result]

		self.state /= np.linalg.norm(self.state)

		if self.DEBUG:
			print(f"STATE {self.state}")

		return result

	def status(self):
		res = ""
		for i in range(2**self.n):
			if i <= 2*self.n:
				res += f"{np.round(self.state[i], 3)}|{self.table[i]}> +"
			else:
				res += f"{np.round(self.state[i], 3)}|{self.table[i]}>"
		print(res)

# Bell state
print("!!! Bell state !!! ")
reg = QubitRegister(False, 2)
reg.H(0)
reg.CX(0, 1)
reg.status()

# Deutsch–Jozsa algorithm
print("!!! Deutsch–Jozsa algorithm!!! ")

# def f(x): return 0
def f(x): return x % 2

reg = QubitRegister(False, 1)
reg.H(0)

# O(f)
i = 0
while i < len(reg.state):
	reg.state[i] *= (-1) ** f(i)
	i += 1

reg.H(0)
print(reg.measure(0))


# Quantum Teleportation
print("!!! Quantum Teleportation !!!")

def teleport_circuit(reg):
	# Prepare shred pair of qubits (1 and 2)
	reg.H(2)
	reg.CX(2, 1)

	# Alice entangles her qubit from entangled pair (1) with qubit to teleport
	reg.CX(0, 1)
	reg.H(0)

	# Alice makes 2 measurements
	m0 = reg.measure(0)
	m1 = reg.measure(1)

	# Bob uses results of measurement to rotate his qubit
	if m1 == 1:
		reg.X(2)
	if m0 == 1:
		reg.Z(2)

	return str(m0), str(m1)

def test_teleport(times, theta):
	measure1_count = 0
	for i in range(times):
		reg = QubitRegister(False, 3)
		reg.Ry(theta, 0)
		t1 = reg.state
		m0, m1 = teleport_circuit(reg)
		t2 = reg.state
		m = reg.measure(2)
		measure1_count += m
		# if t1.all() == t2.all():
		# 	print(f"STATE {i} {colored(m0 + m1 + "1", 'green')}")
		# else:
		# 	print(f"STATE {i} {colored(m0 + m1 + "0", 'red')}")

	return abs(np.sin(0.5*theta)**2 - measure1_count/times)

print(test_teleport(100, 1.234))
print(test_teleport(1000, 1.234))
print(test_teleport(10000, 1.234))
