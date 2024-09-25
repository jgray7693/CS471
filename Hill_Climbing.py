import random

# f(x) used for test cases 1a and 1b
def f(x):
    # f(x) = 2 - x^2
    return 2 - (x**2)

# g(x) used for test case 2
def g(x):
    # g(x) = (0.0051x^5) - (0.1367x^4) + (1.24x^3) - (4.456x^2) + (5.66x) - 0.287
    return (.0051 * (x**5)) - (0.1367 * (x**4)) + (1.24 * (x**3)) - (4.456 * (x**2)) + (5.66 * x) - 0.287



# Hill Climb algorithm using a bounded domain and a predetermined step-size
def hill_climb(func, domain, step, x0):
    # Set current to the starting x-value (x0) and determine the function value at x0
    current = x0
    peak = func(current)
    # While in the bounds of the domain, evaluate the left and right neighbors
    while min(domain) <= current <= max(domain):
        # If the left neighbor is higher than current, move left
        if func(current - step) > peak:
            current = current - step
            peak = func(current)
        # If the right neighbor is higher than current, move right
        elif func(current + step) > peak:
            current = current + step
            peak = func(current)
        # If neither neighbor is higher than current, current yields a local maximum
        else:
            break
    return current, func(current)

# Random restart hill climb algorithm that calls hill climb algorithm at randomly determined intervals to find the absolute maximum of a function in a bounded domain
def random_restart_hill_climb(func, domain, step, num_restarts):
    # Set x_max to the lowest value in the bounded domain, and y_max to the function value at x
    x_max = min(domain)
    y_max = (func(x_max))
    # Split the domain into 4 quadrants. This is to force the random starting points to cover a large a range of values to try to maximize randomness
    quadrants = (max(domain) - min(domain)) // 4
    quadrant_1 = min(domain) + quadrants
    quadrant_2 = quadrant_1 + quadrants
    quadrant_3 = quadrant_2 + quadrants
    # Loop through each quadrant
    for i in range(0,4):
        # In each quadrant, perform num_restarts // 4 hill climb algorithms, with a random integer input in each quadrant
        for x in range(0, num_restarts//4):
            if i == 0:
                x, y = hill_climb(func, domain, step, random.randrange(0,quadrant_1))
            elif i == 1:
                x, y = hill_climb(func, domain, step, random.randrange(quadrant_1, quadrant_2))
            elif i == 2:
                x, y = hill_climb(func, domain, step, random.randrange(quadrant_2, quadrant_3))
            else:
                x, y = hill_climb(func, domain, step, random.randrange(quadrant_3, max(domain)))
            # If the local maximum found from hill_climb is a global_maximum, update x_max and y_max
            if y > y_max:
                y_max = y
                x_max = x
    return x_max, y_max

# Test Cases

# Test Case 1a
####################################################################################################################
x_domain = [-5, 5]                                          # inclusive domain of available inputs
step_size = 0.5                                             # step size for available inputs
start = random.randrange(min(x_domain), max(x_domain), 1)   # random starting-point
x, y = hill_climb(f, x_domain, step_size, start)
print('Test case 1a: f(x) = 2 - x^2')
print(f'Step-size: {step_size}\tx: {round(x,3)}\tMax: {round(y,3)}\n')

# Test Case 1b
####################################################################################################################
x_domain = [-5, 5]                                          # inclusive domain of available inputs
step_size = 0.01                                            # step size for available inputs
start = random.randrange(min(x_domain), max(x_domain), 1)   # random starting-point
x, y = hill_climb(f, x_domain, step_size, start)
print('Test case 1b: f(x) = 2 - x^2')
print(f'Step-size: {step_size}\tx: {round(x,3)}\tMax: {round(y,3)}\n')

# Test Case 2a
####################################################################################################################
x_domain = [0, 10]                                          # inclusive domain of available inputs
step_size = 0.5                                             # step size for available inputs
x, y = random_restart_hill_climb(g, x_domain, step_size, 20)
print('Test case 2a: g(x) = 0.0051x^5 - 0.1367x^4 + 1.24x^3 - 4.456x^2 + 5.66x - 0.287')
print(f'Step-size: {step_size}\tx: {round(x,3)}\tMax: {round(y,3)}\n')

# Test Case 2b
####################################################################################################################
x_domain = [0, 10]                                          # inclusive domain of available inputs
step_size = 0.5                                             # step size for available inputs
x = []                                                      # list to store x-values for random starting points
y = []                                                      # list to store y-values for random starting points
start = []                                                  # list to store random starting points
for i in range (0, 5):
    start_temp = random.randrange(min(x_domain), max(x_domain), 1)   # random starting-point
    x_temp, y_temp = hill_climb(g, x_domain, step_size, start_temp)
    start.append(start_temp)
    x.append(x_temp)
    y.append(y_temp)
print('Test case 2b: g(x) = 0.0051x^5 - 0.1367x^4 + 1.24x^3 - 4.456x^2 + 5.66x - 0.287')
# print start, y, x values for each random starting point
for i in range(0,5):
    print(f'Start: {round(start[i], 3)}\tx: {round(x[i], 3)}\tMax: {round(y[i], 3)}')