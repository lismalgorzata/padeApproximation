import numpy as np
import matplotlib.pyplot as plt

def arctan(x):
    return np.arctan(x)

def pade_approximation(f, n, k, x0):
    """Funkcja obliczająca aproksymację Padé'a dla podanej funkcji f. Funkcja bierze na wejściu 4 argumenty:
    f - funkcja, którą chcemy zaproksymować
    n - parametr określający stopień wielomianu licznika aproksymacji Padé'a
    k - parametr określający stopień wielomianu mianownika aproksymacji Padé'a
    x0 - punkt, w którym obliczane są wartości funkcji f do obliczenia aproksymacji

    Wewnątrz funkcji zostają zdefiniowane macierze C, A, i B, które są wykorzystywane do obliczenia współczynników
    wielomianów licznika i mianownika aproksymacji Padé'a (L i M).
    Funkcja zwraca pary współczynników (L,M)."""

    # Definiowanie macierzy C
    C = np.zeros((n + 1, k + 1))
    for i in range(n + 1):
        for j in range(k + 1):
            if i + j <= n:
                C[i, j] = f(x0 + i + j) / np.math.factorial(i + j)

    # Definiowanie macierzy A i B
    A = np.zeros((n + 1, n + 1))
    B = np.zeros((k + 1, k + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            A[i, j] = C[i, j - i]
        B[0, i] = C[i, k - n + i]
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            B[i, j] = C[n - k + i, j - i]

        # Obliczanie współczynników L i M
    L = np.linalg.lstsq(A, np.ones(n + 1), rcond=None) # A* L = np.ones(n + 1)
    M = np.linalg.lstsq(B, np.ones(k + 1), rcond=None)

    return L[0], M[0]

# Obliczenie aproksymacji wartości n i k
x = np.linspace(0, 5, 100)
results = []
for i in range(10):
    n = i
    k = i
    L, M = pade_approximation(arctan, n, k, 1)
    results.append((L, M))

# Obliczenie wartości aproksymowanej funkcji dla każdej aproksymacji
approximations = []
for L, M in results:
    if np.all(np.abs(L) < 1e-10) or np.all(np.abs(M) < 1e-10):
        continue
    if np.any(M == 0):
        continue
    if np.any(L == 0):
        continue
    approx = np.polyval(L, x) / np.polyval(M, x)
    approximations.append(approx)

# Wykres funkcji aproksymowanej i funkcji interpolacyjnych
plt.plot(x, np.arctan(x), label='arctan(x)', color = "#ff00a6")
for i, (L, M) in enumerate(results):
    # Funkcja enumerate generuje pary (indeks, wartość),
    # gdzie indeks to numer indeksu elementu w tablicy,
    # a wartość to wartość tego elementu.
    if np.any(np.isnan(L)) or np.any(np.isnan(M)):
        continue
    if np.any(np.isnan(np.polyval(M, x))):
        continue
    if np.any(M == 0):
        continue
    if np.any(L == 0):
        continue
    approx = np.polyval(L, x) / np.polyval(M, x)
    if np.any(np.isnan(approx)):
        continue
    plt.plot(x, approx, label=f'R n={i}, k={i}')
plt.legend()
plt.ylim(-15, 15)
plt.show()

# Oszacowanie błędu aproksymacji
errors = []
for approx in approximations:
    x = np.linspace(0, 5, 100)
    error = np.linalg.norm(np.arctan(x) - approx)
    errors.append(error)

# Wyświetlenie błędów aproksymacji
for i, error in enumerate(errors):
    print(f"Błąd aproksymacji n={i}, k={i}: {error:.10f}")
