import math

# Function to determine if two input circles intersect. Circles are of the form (x, y, r)
# Uses math.dist (Euclidean distance) to calculate distance between the center of the two circles
def circles_intersect(circle1, circle2):
    # Get the center of both circles
    center1 = (circle1[0], circle1[1])
    center2 = (circle2[0], circle2[1])
    # If distance between the circles is less than the difference between the radii, the circles are inside one another
    if math.dist(center1, center2) < abs(circle1[2]-circle2[2]):
        return False
    # If distance between the circles is greater than the sum of the radii, the circles are physically separated
    if math.dist(center1, center2) > circle1[2]+circle2[2]:
        return False
    # If neither of the above two conditions exist, the circles overlap on at least one point
    return True

# Takes a list of circles in the form (x, y, r) and processes the list to determine if the circles make a cluster
def evaluate_test(test_case: list):
    # Get the length of the input list, since this list will be modified
    length = len(test_case)
    # Add the first circle in the input list into the cluster_set list and delete it from the input list (this is to save time and space)
    cluster_set = [test_case[0]]
    del test_case[0]
    # Evaluate each circle in the custer_set list with each circle in the input list. If they intersect, remove it from the input list and add it to the cluster_set list
    for circle in cluster_set:
        for unknown_circle in test_case:
            if circles_intersect(circle, unknown_circle):
                cluster_set.append(unknown_circle)
                test_case.remove(unknown_circle)
    # If every circle in the input list has been moved to the cluster_set list -> the lengths match, then the list of circles is a cluster. O   therwise, there is at least one circle that does not intersect.
    return len(cluster_set) == length

# Test Case 1
test_case_1 = [(1,3,0.7), (2,3,0.4), (3,3,0.9)]                         # True
print(f'Test Case 1: {evaluate_test(test_case_1)}')

# Test Case 2
test_case_2 = [(1.5,1.5,1.3), (4,4,0.7)]                                # False
print(f'Test Case 2: {evaluate_test(test_case_2)}')

# Test Case 3
test_case_3 = [(0.5,0.5,0.5), (1.5,1.5,1.1), (0.7,0.7,0.4), (4,4,0.7)]  # False
print(f'Test Case 3: {evaluate_test(test_case_3)}')

# Test Case 4 - My own test case
test_case_4 = [(1,1,4), (1,3,1), (1,5,2)]                               # True
print(f'Test Case 4: {evaluate_test(test_case_4)}')