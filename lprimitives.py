import math
import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares


#############################################################################
# Helper functions for primitive fitting                                    # 
#############################################################################
def mean(points):
    (_, N) = np.shape(points)
    return points * (1 / float(N)) * np.matrix(np.ones((N, 1)))

def find_norms(points):
    (_, N) = np.shape(points)
    return np.matrix([np.linalg.norm(points[:,i]) ** 2 for i in range(0, N)])

def linear_regression(A, b):
    theta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return theta

def is_collinear(points):
    N = len(points)
    if N < 3:
        return True
    for i in range(N - 2):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[i + 1][0], points[i + 1][1]
        x3, y3 = points[i + 2][0], points[i + 2][1]
        a = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if a != 0:
            return False
    return True

def polyfit_regularize(x, y):
    def func(x, a, b, c):
        return a * x ** 2 + b * x + c
    popt_cons, _ = curve_fit(func, x, y, bounds=([-0.4, -100, -np.inf], [0.4, 100, np.inf]))
    return np.array(popt_cons)

def polyfit_regularize_noacc(x, y):
    def func(x, b, c):
        return b * x + c    
    popt_cons, _ = curve_fit(func, x, y, bounds=([-100, -np.inf], [100, np.inf]))
    popt_cons = np.insert(popt_cons, 0, 0, axis=0)
    return np.array(popt_cons)


#############################################################################
# Circular primitive fitting                                                # 
# Fixed & refactored https://github.com/trehansiddharth/fit                 # 
#############################################################################
def circle_fit(orig_points, algorithm='bullock'):
    x_array, y_array = [], []
    for i in range(len(orig_points)):
        x_array.append(orig_points[i][0])
        y_array.append(orig_points[i][1])
    points = np.matrix([x_array, y_array])

    d, N = points.shape
    if algorithm == 'algebraic':
        # Compute the 2-norm of each point in the input
        norms = find_norms(points)

        # Define the matrices A and b we're going to use for optimization
        A = np.hstack((norms.T, points.T))
        b = np.ones((N, 1))
        theta = linear_regression(A, b)

        # Determine the parameters of the algebraic equation of the circle
        a = theta[0].item()
        b = theta[1:]
        c = -1

        # Determine the center and radius
        x = np.matrix(-b / (2.0 * a))
        r = math.sqrt(np.linalg.norm(x) ** 2 + 1 / a)
    else:
        # Transform the coordinates so that they are with respect to the center of mass
        (d, N) = np.shape(points)
        center = mean(points)
        points_c = points - center

        # Compute the norm of every point in the points matrix
        norms = find_norms(points_c)

        # Compute the matrices A and b to use in linear regression
        A = points_c * points_c.T
        b = 0.5 * points_c * norms.T
        theta = linear_regression(A, b)

        # Convert back to unshifted coordinate system and compute radius
        x = theta + center
        r = math.sqrt(np.linalg.norm(theta) ** 2 + np.sum(norms) / float(N))

    x_c, y_c = x.item(0), x.item(1)

    x_m, y_m = orig_points[len(orig_points) // 2][0], orig_points[len(orig_points) // 2][1]

    x1, y1 = orig_points[0][0], orig_points[0][1]
    x2, y2 = orig_points[-1][0], orig_points[-1][1]

    pt1_angle = 180 * np.arctan2(y1 - y_c, x1 - x_c) / np.pi
    pt2_angle = 180 * np.arctan2(y2 - y_c, x2 - x_c) / np.pi
    mid_angle = 180 * np.arctan2(y_m - y_c, x_m - x_c) / np.pi

    angles = []
    quadrants = []
    for idx in range(len(orig_points)):
        x, y = orig_points[idx][0], orig_points[idx][1]
        angle = 180 * np.arctan2(y - y_c, x - x_c) / np.pi
        angles.append(angle)
        quadrants.append(angle // 90)

    counter = 0
    for idx in range(len(angles) - 1):
        if quadrants[idx] == -2 and quadrants[idx + 1] == 1:
            counter -= 360
        if quadrants[idx] == 1 and quadrants[idx + 1] == -2:
            counter += 360
        angles[idx + 1] += counter

    x = np.array(range(len(angles)))
    y = np.array(angles)

    z = polyfit_regularize(x, y)
    p = np.poly1d(z)

    circle_error = 0
    for i in range(len(orig_points)):
        angle = p(i)
        new_x = x_c + r * np.cos(angle * np.pi / 180)
        new_y = y_c + r * np.sin(angle * np.pi / 180)
        if i == 0:
            start_x, start_y = new_x, new_y
        if i == len(orig_points) - 1:
            end_x, end_y = new_x, new_y
        circle_error += np.sqrt((orig_points[i][0] - new_x)**2 + (orig_points[i][1] - new_y)**2)

    span_error = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    return x_c, y_c, r, z, circle_error, span_error

def circle_fit_noacc(orig_points, algorithm='bullock'):
    x_array, y_array = [], []
    for i in range(len(orig_points)):
        x_array.append(orig_points[i][0])
        y_array.append(orig_points[i][1])
    points = np.matrix([x_array, y_array])

    d, N = points.shape
    if algorithm == 'algebraic':
        # Compute the 2-norm of each point in the input
        norms = find_norms(points)

        # Define the matrices A and b we're going to use for optimization
        A = np.hstack((norms.T, points.T))
        b = np.ones((N, 1))
        theta = linear_regression(A, b)

        # Determine the parameters of the algebraic equation of the circle
        a = theta[0].item()
        b = theta[1:]
        c = -1

        # Determine the center and radius
        x = np.matrix(-b / (2.0 * a))
        r = math.sqrt(np.linalg.norm(x) ** 2 + 1 / a)
    else:
        # Transform the coordinates so that they are with respect to the center of mass
        (d, N) = np.shape(points)
        center = mean(points)
        points_c = points - center

        # Compute the norm of every point in the points matrix
        norms = find_norms(points_c)

        # Compute the matrices A and b to use in linear regression
        A = points_c * points_c.T
        b = 0.5 * points_c * norms.T
        theta = linear_regression(A, b)

        # Convert back to unshifted coordinate system and compute radius
        x = theta + center
        r = math.sqrt(np.linalg.norm(theta) ** 2 + np.sum(norms) / float(N))

    x_c, y_c = x.item(0), x.item(1)

    x_m, y_m = orig_points[len(orig_points) // 2][0], orig_points[len(orig_points) // 2][1]

    x1, y1 = orig_points[0][0], orig_points[0][1]
    x2, y2 = orig_points[-1][0], orig_points[-1][1]

    pt1_angle = 180 * np.arctan2(y1 - y_c, x1 - x_c) / np.pi
    pt2_angle = 180 * np.arctan2(y2 - y_c, x2 - x_c) / np.pi
    mid_angle = 180 * np.arctan2(y_m - y_c, x_m - x_c) / np.pi

    angles = []
    quadrants = []
    for idx in range(len(orig_points)):
        x, y = orig_points[idx][0], orig_points[idx][1]
        angle = 180 * np.arctan2(y - y_c, x - x_c) / np.pi
        angles.append(angle)
        quadrants.append(angle // 90)

    counter = 0
    for idx in range(len(angles) - 1):
        if quadrants[idx] == -2 and quadrants[idx + 1] == 1:
            counter -= 360
        if quadrants[idx] == 1 and quadrants[idx + 1] == -2:
            counter += 360
        angles[idx + 1] += counter

    x = np.array(range(len(angles)))
    y = np.array(angles)

    z = polyfit_regularize_noacc(x, y)
    p = np.poly1d(z)

    circle_error = 0
    for i in range(len(orig_points)):
        angle = p(i)
        new_x = x_c + r * np.cos(angle * np.pi / 180)
        new_y = y_c + r * np.sin(angle * np.pi / 180)
        if i == 0:
            start_x, start_y = new_x, new_y
        if i == len(orig_points) - 1:
            end_x, end_y = new_x, new_y
        circle_error += np.sqrt((orig_points[i][0] - new_x)**2 + (orig_points[i][1] - new_y)**2)

    span_error = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    return x_c, y_c, r, z, circle_error, span_error


#############################################################################
# Linear primitive fitting                                                # 
#############################################################################
def linear_fit(points, algorithm='regression'):
    line_error = 0

    ## line fit error
    if algorithm == "endpoints":
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[-1][0], points[-1][1]

        m, c = None, None
        if x1 == x2:
            for i in range(len(points)):
                line_error += (x1 - points[i][0]) ** 2
        else:
            m = (y2 - y1) / (x2 - x1)
            c = (y1 * x2 - y2 * x1) / (x2 - x1)
            for i in range(len(points)):
                line_error += (points[i][1] - m * points[i][0] - c) ** 2 / (1 + m**2)

        return m, c, line_error

    elif algorithm == "regression":
        X = [point[0] for point in points]
        Y = [point[1] for point in points]
        N = len(X)
        meanx = sum(X) / len(X)
        meany = sum(Y) / len(Y)
        denom = sum((xi - meanx)**2 for xi in X)

        m, c = None, None

        if denom == 0:
            distances = []
            for idx in range(N):
                distances.append(Y[idx] - Y[0])

            x = np.array(range(N))
            ad = polyfit_regularize(x, np.array(distances))
            ax = np.array([0, 0, X[0]])
            ay = np.array([ad[0], ad[1], ad[2] + Y[0]])

        else:
            numer = sum((xi - meanx) * (yi - meany) for xi, yi in zip(X, Y))
            m = numer / denom
            c = meany - m * meanx

            x_start, y_start = None, None
            distances = []
            for idx in range(N):
                x_proj = (m * Y[idx] - m * c + X[idx]) / (1 + m**2)
                y_proj = (m**2 * Y[idx] + m * X[idx] + c) / (1 + m**2)
                if idx == 0:
                    x_start, y_start = x_proj, y_proj
                d = np.sqrt((x_proj - x_start) ** 2 + (y_proj - y_start) ** 2)
                distances.append(d * np.sign(x_proj - x_start))

            x = np.array(range(N))
            ad = polyfit_regularize(x, np.array(distances))

            angle = np.arctan2(m, 1)
            cos, sin = np.cos(angle), np.sin(angle)
            ax = np.array([ad[0] * cos, ad[1] * cos, ad[2] * cos + x_start])
            ay = np.array([ad[0] * sin, ad[1] * sin, ad[2] * sin + y_start])

        line_error = 0
        x_f, y_f = np.poly1d(ax), np.poly1d(ay)
        for i in range(len(points)):
            if i == 0:
                start_x, start_y = x_f(i), y_f(i)
            if i == len(points) - 1:
                end_x, end_y = x_f(i), y_f(i)
            line_error += np.sqrt((X[i] - x_f(i)) ** 2 + (Y[i] - y_f(i)) ** 2)

        span_error = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        return ax, ay, line_error, span_error


def linear_fit_noacc(points):
    line_error = 0
    X = [point[0] for point in points]
    Y = [point[1] for point in points]
    N = len(X)
    meanx = sum(X) / len(X)
    meany = sum(Y) / len(Y)
    denom = sum((xi - meanx)**2 for xi in X)

    m, c = None, None

    if denom == 0:
        distances = []
        for idx in range(N):
            distances.append(Y[idx] - Y[0])

        x = np.array(range(N))
        ad = polyfit_regularize_noacc(x, np.array(distances))
        ax = np.array([0, 0, X[0]])
        ay = np.array([ad[0], ad[1], ad[2] + Y[0]])

    else:
        numer = sum((xi - meanx) * (yi - meany) for xi, yi in zip(X, Y))
        m = numer / denom
        c = meany - m * meanx

        x_start, y_start = None, None
        distances = []
        for idx in range(N):
            x_proj = (m * Y[idx] - m * c + X[idx]) / (1 + m**2)
            y_proj = (m**2 * Y[idx] + m * X[idx] + c) / (1 + m**2)
            if idx == 0:
                x_start, y_start = x_proj, y_proj
            d = np.sqrt((x_proj - x_start) ** 2 + (y_proj - y_start) ** 2)
            distances.append(d * np.sign(x_proj - x_start))

        x = np.array(range(N))
        ad = polyfit_regularize_noacc(x, np.array(distances))

        angle = np.arctan2(m, 1)
        cos, sin = np.cos(angle), np.sin(angle)
        ax = np.array([ad[0] * cos, ad[1] * cos, ad[2] * cos + x_start])
        ay = np.array([ad[0] * sin, ad[1] * sin, ad[2] * sin + y_start])

    line_error = 0
    x_f, y_f = np.poly1d(ax), np.poly1d(ay)
    for i in range(len(points)):
        if i == 0:
            start_x, start_y = x_f(i), y_f(i)
        if i == len(points) - 1:
            end_x, end_y = x_f(i), y_f(i)
        line_error += np.sqrt((X[i] - x_f(i)) ** 2 + (Y[i] - y_f(i)) ** 2)

    span_error = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    return ax, ay, line_error, span_error


#############################################################################
# Stationary primitive fitting                                              # 
#############################################################################
def stationary_fit(points, algorithm='mean'):
    if algorithm == "mean":
        x_mean, y_mean = 0, 0
        curr_error = 0

        for i in range(len(points)):
            x_mean += points[i][0]
            y_mean += points[i][1]
        x_mean /= len(points)
        y_mean /= len(points)

        for i in range(len(points)):
            curr_error += np.sqrt((points[i][0] - x_mean) ** 2 + (points[i][1] - y_mean) ** 2)

        return x_mean, y_mean, curr_error


#############################################################################
# Check if all points lie within a small patch, mark stationary             # 
#############################################################################
def is_localized(points, tol=10):
    min_x, min_y, max_x, max_y = 1000, 1000, 0, 0
    for point in points:
        min_x = point[0] if point[0] < min_x else min_x
        min_y = point[1] if point[1] < min_y else min_y
        max_x = point[0] if point[0] > max_x else max_x
        max_y = point[1] if point[1] > max_y else max_y
    is_localized = False
    if max_x - min_x < tol and max_y - min_y < tol:
        is_localized = True
    return is_localized
