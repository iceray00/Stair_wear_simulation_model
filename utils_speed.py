import math
import random
import numpy as np
from pygments.lexers import go


def calculate_mu_mean_speed(wear_elements):
    """Calculate the mean friction coefficient for the wear elements"""
    mu_mean = np.mean([element.mu for element in wear_elements])
    return round(mu_mean, 3)  # 保留4位小数


def calculate_wear_volume_speed(k, F, s, H, show=False):
    """Calculate the wear volume 'V' based on the Archard wear model"""
    if show: print(f"in calculate_wear_volume, k: {k}, F: {F}, s: {s}, H: {H}")
    V = k * F * s / H
    return round(V, 4)  # 保留4位小数


def calculate_sliding_displacement_speed(mu_mean, theta, v, g, show=False):
    """Calculate the relative sliding displacement 's'"""
    if show: print(f"in calculate_sliding_displacement, mu_mean: {mu_mean}, theta: {theta}, v: {v}, g: {g}")
    # in m
    s = 0.18 * v ** 2 * (math.cos(theta) ** 2) / (mu_mean * g)
    # in mm
    s *= 1000
    return round(s, 4)  # 保留4位小数


def calculate_sliding_displacement_speed_fast(mu_mean, theta_value, v, g, show=False):
    """Calculate the relative sliding displacement 's'"""
    if show: print(f"in calculate_sliding_displacement, mu_mean: {mu_mean}, theta_value: {theta_value}, v: {v}, g: {g}")
    # in m
    s = 0.18 * v ** 2 * (theta_value ** 2) / (mu_mean * g)
    # in mm
    s *= 1000
    return round(s, 4)  # 保留4位小数


def generate_p(length, width, feet):
    """Generate a random point P within the given range, using normal distribution"""
    P_x = random.gauss(length / 2, 0.2 * length)  # Normal distribution for x
    P_y = random.gauss(width / 2, 0.2 * width)    # Normal distribution for y

    # Ensure the point lies within the boundaries
    P_x = min(max(P_x, 0.2 * length), 0.8 * length)
    P_y = min(max(P_y, 0.2 * width), 0.8 * width)
    return P_x, P_y


def calculate_mu_mean(wear_elements):
    """Calculate the mean friction coefficient for the wear elements"""
    mu_mean = np.mean([element.mu for element in wear_elements])
    return mu_mean


def calculate_sliding_displacement(mu_mean, theta, v, g, show=False):
    """Calculate the relative sliding displacement 's'"""
    if show: print(f"in calculate_sliding_displacement, mu_mean: {mu_mean}, theta: {theta}, v: {v}, g: {g}")
    # in m
    s = 0.18 * v**2 * (math.cos(theta)**2) / (mu_mean * g)
    # in mm
    s *= 1000
    return s


def calculate_wear_volume(k, F, s, H, show=False):
    """Calculate the wear volume 'V' based on the Archard wear model"""
    if show: print(f"in calculate_wear_volume, k: {k}, F: {F}, s: {s}, H: {H}")
    V = k * F * s / H
    return V


def get_wear_elements_up(surface_elements, P_x, P_y, r, c):
    """Generate the wear elements based on the shoe model"""
    wear_elements = []

    for element in surface_elements:
        distance = math.sqrt((P_x - element.x) ** 2 + (P_y - element.y) ** 2)

        # Check if the point is inside the shoe model
        # Check if in the semicircle
        if distance <= r:
            wear_elements.append(element)
        # Check if inside the rectangle
        elif P_y - c <= element.y <= P_y and P_x - r <= element.x <= P_x + r:
            wear_elements.append(element)

    return wear_elements


def get_wear_elements_down(surface_elements, P_x, P_y, r, c):
    """Generate the wear elements based on the shoe model. When Down"""
    wear_elements = []

    for element in surface_elements:
        distance = math.sqrt((P_x - element.x) ** 2 + (P_y - element.y) ** 2)

        # Check if the point is inside the shoe model
        # Check if in the semicircle
        if distance <= r:
            wear_elements.append(element)
        # Check if inside the rectangle
        elif P_y <= element.y <= P_y + c and P_x - r <= element.x <= P_x + r:
            wear_elements.append(element)

    return wear_elements


def get_wear_elements_hybrid(surface_elements, P_x, P_y, r, c, flag_up_down):
    if flag_up_down == 'up':
        return get_wear_elements_up(surface_elements, P_x, P_y, r, c)
    elif flag_up_down == 'down':
        return get_wear_elements_down(surface_elements, P_x, P_y, r, c)
    else:
        raise ValueError("flag_up_down must be 'up' or 'down'")




# /* ================= Save surface elements to JSON ================= */

import json


def element_to_dict(element):
    """
    Convert a single FiniteElement object to a dictionary.
    """
    return {
        'x': element.x,
        'y': element.y,
        'h': element.h,
        'hp': float(element.hp),  # 转换为Python原生浮点数
        'mu': float(element.mu)  # 转换为Python原生浮点数
    }


def save_surface_elements_to_json(surface_elements, filename='surface_elements.json'):
    """
    Save the surface elements to a JSON file.
    """
    # 将所有元素转换为字典列表
    elements_dict = [element_to_dict(element) for element in surface_elements]

    # 将字典列表写入JSON文件
    with open(filename, 'w') as f:
        json.dump(elements_dict, f, indent=4)
    print(f"Surface elements saved to {filename}")









