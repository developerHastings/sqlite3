def numerical_derivative(f, x, dx=1e-5):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

def f(x):
    return 3 * x**3 + 2 * x

x = 1
dx_values = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11]

print(f"True derivative at x={x} is 11")  # Because f'(x) = 9x^2 + 2 and 9*1 + 2 = 11

for dx in dx_values:
    approx = numerical_derivative(f, x, dx)
    print(f"dx = {dx:.0e} => approx derivative = {approx:.10f}")
