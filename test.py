import math
from sympy import isprime, legendre_symbol, gcd
import numpy as np
# ------------------------
# Step 1: Base-m polynomial selection
# ------------------------
def base_m_expansion(n, m, degree):
    """Expand n in base m up to the given degree."""
    coeffs = []
    temp = n
    for _ in range(degree + 1):
        coeffs.append(temp % m)
        temp //= m
    return coeffs[::-1]  # highest-degree first

def construct_polynomial_from_m(n, m, degree):
    """Construct polynomial f(x) such that f(m) = n."""
    coeffs = base_m_expansion(n, m, degree)
    return coeffs

def evaluate_polynomial(coeffs, x):
    """Evaluate polynomial at x."""
    result = 0
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        result += c * (x ** (deg - i))
    return result

def polynomial_function(coeffs):
    """Return a lambda function f(x) from coefficients."""
    return lambda x: evaluate_polynomial(coeffs, x)

# ------------------------
# Step 2: Rational factor base R
# ------------------------
def rational_factor_base(bound):
    return [p for p in range(2, bound+1) if isprime(p)]

def factorize_over_base(x, base):
    exponents = []
    for p in base:
        count = 0
        while x % p == 0:
            x //= p
            count += 1
        exponents.append(count)
    return exponents, x  # remaining cofactor

# ------------------------
# Step 3: Algebraic factor base A
# ------------------------
def algebraic_factor_base(f, primes):
    A = []
    for p in primes:
        for r in range(p):
            if f(r) % p == 0:
                A.append((r, p))
    return A
# ------------------------
# Step 4: Quadratic character base Q
# ------------------------
def quadratic_character_base(f, primes, exclude_A):
    Q = []
    for q in primes:
        if q in exclude_A:
            continue
        for s in range(q):
            if f(s) % q == 0:
                Q.append((s, q))
    return Q
# ------------------------
# Step 5: Find smooth pairs (a, b)
# ------------------------
def find_smooth_pairs(n, m, f, R, A, Q, a_limit=20, b_limit=3):
    """
    Find smooth pairs for small n for teaching GNFS.
    - Fully factorizes over R and A (small factor bases)
    - Early rejection
    - Small a and b ranges to keep it fast
    """
    pairs = []
    A_primes = [p for _, p in A]

    for b in range(1, b_limit):
        for a in range(-a_limit, a_limit):
            # Rational side
            val_r = abs(a + b*m)
            rat_exp, rem_r = factorize_over_base(val_r, R)
            if rem_r != 1:
                continue  # Not fully smooth

            # Algebraic side
            val_a = abs(a + b*f(m))
            alg_exp, rem_a = factorize_over_base(val_a, A_primes)
            if rem_a != 1:
                continue  # Not fully smooth

            # Quadratic characters
            quad_exp = [0 if legendre_symbol(a + b*s, q) == 1 else 1 for s, q in Q]

            # Store the pair
            pairs.append({
                "a": a,
                "b": b,
                "rational": rat_exp,
                "algebraic": alg_exp,
                "quadratic": quad_exp,
                "sign": 0 if (a + b*m) >= 0 else 1
            })

    return pairs
# ------------------------
# Step 6: Construct matrix X
# ------------------------
def construct_matrix_X(pairs, R_len, A_len, Q_len):
    matrix = []
    for p in pairs:
        row = [p["sign"]] + p["rational"] + p["algebraic"] + p["quadratic"]
        matrix.append([x % 2 for x in row])
    return np.array(matrix, dtype=int)
# ------------------------
# Step 7: Solve linear system modulo 2
# ------------------------
def solve_linear_system_mod2(X):
    mat = X.copy()
    rows, cols = mat.shape
    pivot_rows = []
    dependencies = []

    for col in range(cols):
        pivot_row = None
        for r in range(rows):
            if r in pivot_rows:
                continue
            if mat[r, col] == 1:
                pivot_row = r
                pivot_rows.append(r)
                break
        if pivot_row is None:
            continue
        for r in range(rows):
            if r != pivot_row and mat[r, col] == 1:
                mat[r] = (mat[r] + mat[pivot_row]) % 2

    for i, row in enumerate(mat):
        if not row.any():
            dependencies.append(i)
    return dependencies if dependencies else [0]
# ------------------------
# Step 8: Factor n using difference of squares
# ------------------------
def difference_of_squares_factor(n, s, r):
    factor1 = gcd(s + r, n)
    factor2 = gcd(s - r, n)
    factors = [f for f in (factor1, factor2) if 1 < f < n]
    return factors[0] if factors else None

def factor_n_from_pairs(n, pairs, dependency, m, f):
    x_prod = 1
    y_prod = 1
    for idx in dependency:
        a = pairs[idx]["a"]
        b = pairs[idx]["b"]
        x_prod *= a + b*m
        y_prod *= a + b*f(m)
    factor = difference_of_squares_factor(n, int(math.isqrt(y_prod)), x_prod)
    return factor
def polynomial_to_string(coeffs):
    """Convert a list of coefficients into a polynomial string. Only for print purposes"""
    degree = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if c == 0:
            continue
        if power == 0:
            terms.append(f"{c}")
        elif power == 1:
            terms.append(f"{c}*x")
        else:
            terms.append(f"{c}*x**{power}")
    return " + ".join(terms)
def gnfs(n, m, degree=3, R_bound=30, A_bound=90, Q_bound=107):
    # Step 1
    coeffs = construct_polynomial_from_m(n, m, degree)
    f = polynomial_function(coeffs)
    print(f"Polynomial f(x) = {polynomial_to_string(coeffs)}")

    # Step 2
    R = rational_factor_base(R_bound)
    print(f"Rational factor base R: {R}")

    # Step 3
    A_primes = [p for p in range(2, A_bound) if isprime(p)]
    A = algebraic_factor_base(f, A_primes)
    print(f"Algebraic factor base A: {A}")

    # Step 4
    Q_primes = [p for p in range(A_bound+1, Q_bound+1) if isprime(p)]
    Q = quadratic_character_base(f, Q_primes, [p for _, p in A])
    print(f"Quadratic character base Q: {Q}")

    # Step 5
    pairs = find_smooth_pairs(n, m, f, R, A, Q)
    print(f"Found {len(pairs)} smooth pairs")

    if len(pairs) == 0:
        print("Not enough smooth pairs. Increase limits or bounds.")
        return None

    # Step 6
    X = construct_matrix_X(pairs, len(R), len(A), len(Q))

    # Step 7
    dependency = solve_linear_system_mod2(X)

    # Step 8
    factor = factor_n_from_pairs(n, pairs, dependency, m, f)
    return factor

n = 221
m = 5
degree = 2
factor = gnfs(n, m, degree)
if factor:
    print(f"{n} = {factor} * {n // factor}")
else:
    print(f"Could not factor {n}")