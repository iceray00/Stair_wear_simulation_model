import argparse
import random
import numpy as np
import time
import math

from matplotlib.colors import Normalize
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.stats import norm
from scipy.optimize import fsolve

# speed utils
from utils_speed import calculate_sliding_displacement_speed, calculate_wear_volume_speed, calculate_mu_mean_speed, \
    calculate_sliding_displacement_speed_fast, get_wear_elements_up, generate_p, get_wear_elements_down, \
    get_wear_elements_hybrid, save_surface_elements_to_json

from visual import visualize_surface_pro_max_plotly, visualize_surface_pro_max_surface, visualize_surface_pro_max


np.set_printoptions(precision=4)
np.float = np.float32  # Use 32-bit floating-point numbers instead of the default 64 bits


class FiniteElement:
    """
    Represents a single finite element (or particle) in the 3D stair model.
    Each element corresponds to a 1cm³ block.
    每个粒子的HP取值[0, 1000]，1 hp = 1mm³
    """
    def __init__(self, x, y, h, mu_0, mu_min):
        # self.hp = 1000     # Initial "health", indicating the full unwearable unit
        # self.hp = np.float32(1000)  # 使用32位浮点数
        self.hp = np.int32(1000)

        self.x = x         # x coordinate (side to side)
        self.y = y         # y coordinate (front to back)
        self.h = h         # h coordinate (depth from the surface)

        # self.mu = self.calculate_mu(mu_0, mu_min, h)  # Initial friction coefficient based on depth h
        self.mu = np.float32(self.calculate_mu(mu_0, mu_min, h))


    def calculate_mu(self, mu_0, mu_min, h):
        """
        Calculate the friction coefficient (mu) based on the depth h.
        Assuming an exponential decay from mu_0 to mu_min with increasing h.
        """
        mu = (mu_0 - mu_min) * np.exp(-h / 2) + mu_min  # Example exponential model for mu
        # return mu
        return round(mu, 4)

    def show_elements(self):
        """
        Display the elements in the sandbox.
        """
        print(f"Element at ({self.x}, {self.y}, {self.h}): hp = {self.hp}, mu = {self.mu}")



class SandBox:
    """
    Represents the entire 3D stair sandbox, composed of many finite elements (particles).
    The sandbox is a rectangular block with specified length, width, and height.
    """
    def __init__(self, length, width, height, mu_0, mu_min, theta, materials, k, F, H):
        self.length = length  # Length of the stair in cm
        self.width = width    # Width of the stair in cm
        self.height = height  # Height (depth) of the stair in cm
        self.mu_0 = mu_0      # Surface friction coefficient
        self.mu_min = mu_min  # Minimum friction coefficient at depth
        self.theta = theta    # Angle of whole stair in degrees
        self.materials = materials  # Type of materials
        self.k = k            # Wear coefficient of materials
        self.F = F            # Normal Load in N
        self.H = H            # Hardness in MPa
        self.g = 9.81         # Acceleration due to gravity

        self.all_elements = self.create_elements()
        self.surface_elements = self.get_surface_elements()

    def create_elements(self):
        """
        Create the 3D grid of finite element particles representing the sandbox.
        Each element is a 1mm³ block.
        """
        all_elements = []
        for x in tqdm(range(self.length), desc="Creating elements, X ", unit="cm"):
            for y in range(self.width):
                for h in range(self.height):
                    # Create each finite element with x, y, h positions and associated mu
                    # if h >= self.height * (2/3) and y >= self.width * (2/3) :
                    #     continue
                    element = FiniteElement(x, y, h, self.mu_0, self.mu_min)
                    all_elements.append(element)
        return all_elements

    def get_surface_elements(self):
        """
        Generate a list of surface elements. For each (x, y) coordinate,
        find the first element (smallest h) where hp > 0.
        """
        surface_elements = []

        for x in range(self.length):
            for y in range(self.width):
                # Find the smallest h where hp > 0 for this (x, y) coordinate
                for h in range(self.height):
                    element = self.get_element(x, y, h)
                    if element.hp > 0:  # We only care about elements that are not fully worn
                        surface_elements.append(element)
                        break  # Stop at the first (smallest h) element where hp > 0
        return surface_elements

    def get_element(self, x, y, h, show=False):
        """
        Get the finite element at specific coordinates.
        """
        start_time = time.time()  # Start time measurement
        index = (x * self.width * self.height) + (y * self.height) + h
        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        if show: print(f"Function 'get_element' took {execution_time:.6f} seconds to execute.")

        return self.all_elements[index]

    def display_info(self):
        """
        Display the basic information of the sandbox (size and number of elements).
        """
        print(f"Sandbox size: {self.length}cm x {self.width}cm x {self.height}cm")
        print(f"Total elements: {len(self.all_elements)}")

    def visualize_surface(self):
        start_time = time.time()  # Start time measurement
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Prepare lists for surface plotting
        x_vals = []
        y_vals = []
        h_vals = []  # Depth will be the Z-axis (h-axis)
        mu_vals = []  # Friction coefficient (mu) for color mapping

        for x in range(self.length):
            for y in range(self.width):
                for h in range(self.height):
                    # Only consider the surface elements (e.g., h=0 or h=height-1 or edge of x/y)
                    if x == 0 or x == self.length - 1 or y == 0 or y == self.width - 1 or h == 0 or h == self.height - 1:
                        element = self.get_element(x, y, h)
                        if element.hp > 0:  # Only plot elements that are not fully worn
                            x_vals.append(x)
                            y_vals.append(y)
                            h_vals.append(h)  # Depth will be plotted on the Z-axis (h_vals)
                            mu_vals.append(element.mu)  # Use mu for coloring

        for element in self.surface_elements:
            x_vals.append(element.x)
            y_vals.append(element.y)
            h_vals.append(element.h)  # Depth will be plotted on the Z-axis
            mu_vals.append(element.mu)  # Use mu for coloring

        # Customize the color gradient
        colors = ["red", "yellow", "green", "blue"]
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_yellow_red", colors)

        # Plot the surface using mu as color mapping

        # cmap = 'cividis_r'
        scatter = ax.scatter(x_vals, y_vals, h_vals, c=mu_vals, cmap=cmap, s=1, alpha=1, norm = Normalize(vmin=self.mu_min, vmax=self.mu_0))


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth (H)')  # Use H as the depth, which is the negative direction
        ax.set_title('Visualization of Surface Wear')

        # Add color bar to indicate the friction coefficient (mu)
        fig.colorbar(scatter, ax=ax, label='Friction Coefficient (mu)')

        # Invert the Z-axis to make depth go from top to bottom
        ax.invert_zaxis()

        plt.show()

        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        print(f"Color Mapping is: {cmap.name}")
        print(f"Function 'visualize_surface' took {execution_time:.6f} seconds to execute.")


    def visualize_surface_from_elements(self):
        start_time = time.time()  # Start time measurement
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Prepare lists for surface plotting
        x_vals = []
        y_vals = []
        h_vals = []  # Depth will be the Z-axis (h-axis)
        mu_vals = []  # Friction coefficient (mu) for color mapping

        # Iterate over surface_elements, which contains the non-worn elements
        for element in self.surface_elements:
            x_vals.append(element.x)
            y_vals.append(element.y)
            h_vals.append(element.h)  # Depth will be plotted on the Z-axis
            mu_vals.append(element.mu)  # Use mu for coloring

        # Custom color gradient
        colors = ["red", "yellow", "green", "blue"]
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_yellow_red", colors)

        # Plot the surface using mu as color mapping
        scatter = ax.scatter(x_vals, y_vals, h_vals, c=mu_vals, cmap=cmap, s=1, alpha=1,
                             norm=Normalize(vmin=self.mu_min, vmax=self.mu_0))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Depth (H)')  # Use H as the depth, which is the negative direction
        ax.set_title('Visualization of Surface Wear')

        # Add color bar to indicate the friction coefficient (mu)
        fig.colorbar(scatter, ax=ax, label='Friction Coefficient (mu)')

        # Invert the Z-axis to make depth go from top to bottom
        ax.invert_zaxis()

        plt.savefig('3d_surface.pdf')
        plt.show()

        end_time = time.time()  # End time measurement
        execution_time = end_time - start_time
        print(f"Color Mapping is: {cmap.name}")
        print(f"Function 'visualize_surface_from_elements' took {execution_time:.6f} seconds to execute.")


def visualize_surface_plus_layer(Sandbox, caption_left='50', caption_right='50'):
    start_time = time.time()  # Start time measurement
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare lists for surface plotting
    x_vals = []
    y_vals = []
    h_vals = []  # Depth will be the Z-axis (h-axis)
    mu_vals = []  # Friction coefficient (mu) for color mapping

    for x in range(Sandbox.length):
        for y in range(Sandbox.width):
            for h in range(Sandbox.height):
                # Only consider the surface elements (e.g., h=0 or h=height-1 or edge of x/y)
                if x == 0 or x == Sandbox.length - 1 or y == 0 or y == Sandbox.width - 1 or h == 0 or h == Sandbox.height - 1:
                    element = Sandbox.get_element(x, y, h)
                    if element.hp > 0:  # Only plot elements that are not fully worn
                        x_vals.append(x)
                        y_vals.append(y)
                        h_vals.append(h)  # Depth will be plotted on the Z-axis (h_vals)
                        mu_vals.append(element.mu)  # Use mu for coloring

    for element in Sandbox.surface_elements:
        x_vals.append(element.x)
        y_vals.append(element.y)
        h_vals.append(element.h)  # Depth will be plotted on the Z-axis
        mu_vals.append(element.mu)  # Use mu for coloring

    # 自定义渐变颜色
    colors = ["red", "yellow", "green", "blue"]
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_yellow_red", colors)

    # Plot the surface using mu as color mapping

    # cmap = 'cividis_r'
    scatter = ax.scatter(x_vals, y_vals, h_vals, c=mu_vals, cmap=cmap, s=1, alpha=1, norm = Normalize(vmin=Sandbox.mu_min, vmax=Sandbox.mu_0))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth (H)')  # Use H as the depth, which is the negative direction
    ax.set_title('Visualization of Surface Wear')

    # Add color bar to indicate the friction coefficient (mu)
    fig.colorbar(scatter, ax=ax, label='Friction Coefficient (mu)')

    # Invert the Z-axis to make depth go from top to bottom
    ax.invert_zaxis()

    plt.savefig('surface_layer.pdf')

    plt.show()

    end_time = time.time()  # End time measurement
    execution_time = end_time - start_time
    print(f"Color Mapping is: {cmap.name}")
    print(f"Function 'visualize_surface' took {execution_time:.6f} seconds to execute.")



# /* ============================== Main ================================ */

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--length', type=int, default=150, help='Length of the stairs, in cm. Default = 150')
    parser.add_argument('-w','--width', type=int, default=30, help='Width of the stairs, in cm. Default = 30')
    parser.add_argument('--height', type=int, default=15, help='Height of the stairs, in cm. Default = 15')
    parser.add_argument('--mu_0', type=float, default=0.6, help='Surface friction coefficient. Default = 0.5')
    parser.add_argument('--mu_min', type=float, default=0.5, help='Minimum friction coefficient at depth. Default = 0.4')
    parser.add_argument('--theta', type=float, default=26.566, help='Angle of the stairs in degrees. Default = 26.566. Meaning tan(26.566) = 1/2')
    parser.add_argument('-v','--v', type=float, default=0.8, help='Speed of the people who walking the stairs. Default = 0.6')

    parser.add_argument('--materials', type=str, default='moorstone', help='Material Name of the stairs')
    parser.add_argument('-k','--k', type=float, default=10**(-4), help='Wear coefficient. Default = 10**(-4)')
    parser.add_argument('-F','--F', type=float, default=600, help='Normal load of the stairs, in N. Default = 600')
    parser.add_argument('-H','--H', type=float, default=200, help='Hardness of the stairs, in MPa. Default = 200')

    parser.add_argument('--silent', action='store_true', default=True, help='Silent mode')
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize the stairs')
    parser.add_argument('--up_down_radio', type=float, default=0.5, help='Up and down ratio of the stairs. `0.2` meaning 20 percent up and 80 percent down. 0 <= up_down_radio <= 1. Default = 0.5')
    parser.add_argument('--single_sideBySide_radio', type=float, default=0.5, help='Probability of walking side by side (0 = all walking alone). 0 <= single_sideBySide_radio <= 1. Default = 0.5')
    parser.add_argument('--stair_use_per_day', type=int, default=19000, help='Number of stairs used per day. Default = 10000')
    parser.add_argument('--use_year', type=int, default=400, help='Number of years. Default = 200')
    parser.add_argument('--aggregate',type=int , default=1000000, help='Number of stairs aggregated. Default = 1000000')

    args = parser.parse_args()


    # in cm
    length = args.length
    width = args.width
    height = args.height

    mu_0 = args.mu_0
    mu_min = args.mu_min

    theta = math.pi / 180 * args.theta  # make theta in radians
    v = args.v

    # The wear coefficient k and hardness H can be found in the relevant database
    materials = args.materials
    k = args.k
    F = args.F
    H = args.H

    # r = 46.422   # radius of the semicircle in mm
    r = 4.6422   # radius of the semicircle in cm

    # c = 243.578  # length of the rectangle in mm
    c = 24.3578  # length of the rectangle in cm

    g = 9.81     # acceleration due to gravity in m/s^2

    silent = args.silent
    visualize = args.visualize


    # Percentage going up and down stairs (0 = all going down stairs)
    up_down_radio = args.up_down_radio
    # probability of walking side by side (0 = all walking alone)
    single_sideBS = args.single_sideBySide_radio

    frequency_of_stair_use_during_1_day = args.stair_use_per_day

    one_year = 365
    use_year = args.use_year
    total_day = one_year * use_year

    # Total use of the stairs
    total_step = frequency_of_stair_use_during_1_day * total_day

    use_aggregate = args.aggregate

    # Treat the aggregated data as an epoch
    # i.e., subsequent Monte Carlo simulation, the wear volume in each epoch is the result of use_aggregate so many people stepping on it
    F *= use_aggregate

    epoches = total_step // use_aggregate
    total_step = epoches * use_aggregate

    if silent:
        print(f"\n    @@@@ Total steps: {total_step} @@@@")
        print(f"    @@@@ Stair Use a Day: {frequency_of_stair_use_during_1_day} @@@@\n")
        print(f"    @@@@ Use Year: {use_year} @@@@\n")
        print(f"    @@@@ Use Aggregate: {use_aggregate} @@@@\n")
        print(f"    @@@@ Epoches: {epoches} @@@@\n")


    # in cm
    center = length / 2

    def equation_v2(sigma):
        z1 = center / sigma  # Leave 10cm on both sides
        z2 = - (center - 20) / sigma
        # Calculate the difference in CDF
        return norm.cdf(z1) - norm.cdf(z2) - 0.95

    # Solve with fsolve
    sigma_solution_v2 = fsolve(equation_v2, length / 5)
    sigma_solution = sigma_solution_v2[0]
    print(sigma_solution)

    def generate_p_single_up(length, width, left_or_right):
        """Generate a random point P within the given range, using normal distribution"""
        center = length / 2

        # Determine the mean based on left_or_right
        if left_or_right == 'left':
            mean_x = center - 15  # Mean on the left
        elif left_or_right == 'right':
            mean_x = center + 15  # Mean on the right
        else:
            raise ValueError("left_or_right must be 0 (left) or 1 (right)")

        while True:
            # generate P_x using normal distribution
            P_x = random.gauss(mean_x, sigma_solution)

            # Generate P_y using a gamma distribution
            shape_y = 3  # The shape parameter k of the gamma distribution
            scale_y = 0.1 * width  # Scaling parameter θ for the gamma distribution
            P_y = np.random.gamma(shape_y, scale_y)  # gamma分布的缩放参数θ

            # check if the boundary condition is satisfied
            if 5 <= P_x <= length - 5 and 5 <= P_y <= width - 5:
                break
        return P_x, P_y

    def generate_p_single_down(length, width, left_or_right):
        """Generate a random point P within the given range, using normal distribution"""
        center = length / 2

        # Determine the mean based on left_or_right
        if left_or_right == 'left':
            mean_x = center - 15
        elif left_or_right == 'right':
            mean_x = center + 15
        else:
            raise ValueError("left_or_right must be 0 (left) or 1 (right)")

        while True:
            P_x = random.gauss(mean_x, sigma_solution)

            # generate P_y using gamma distribution and map to interval [0, length-c]
            shape_y = 3
            scale_y = 0.05 * width
            P_y_gamma = np.random.gamma(shape_y, scale_y)

            # map to [0, length-c]
            P_y = (width - (c)) - P_y_gamma

            if 5 <= P_x <= length - 5 and 0 <= P_y <= width - (c):
                break

        return P_x, P_y

    def generate_p_single_hybrid(length, width, flag_feet, up_down_radio=0.5):
        """Generate a random point P within the given range, using normal distribution"""
        random_f = random.random()

        if random_f < up_down_radio:
            return *generate_p_single_up(length, width, flag_feet), 'up'
        else:
            return *generate_p_single_down(length, width, flag_feet), 'down'

    def generate_p_up_left(length, width, left_or_right):
        """Generate a random point P within the given range, using normal distribution"""
        center = length / 2

        if left_or_right == 'left':
            mean_x_left = center / 2 + 2 - 10
        elif left_or_right == 'right':
            mean_x_left = center / 2 + 2 + 10
        else:
            raise ValueError("left_or_right must be 0 (left) or 1 (right)")

        while True:

            P_x = random.gauss(mean_x_left, sigma_solution)

            shape_y = 3
            scale_y = 0.1 * width
            P_y = np.random.gamma(shape_y, scale_y)

            if 5 <= P_x <= center and 4 <= P_y <= width - 5:
                break
        return P_x, P_y

    def generate_p_up_right(length, width, left_or_right):
        """Generate a random point P within the given range, using normal distribution"""
        center = length / 2

        if left_or_right == 'left':
            mean_x_right = length - center / 2 - 2 - 10
        elif left_or_right == 'right':
            mean_x_right = length - center / 2 - 2 + 10
        else:
            raise ValueError("left_or_right must be 0 (left) or 1 (right)")

        while True:
            P_x = random.gauss(mean_x_right, sigma_solution)

            shape_y = 3
            scale_y = 0.1 * width
            P_y = np.random.gamma(shape_y, scale_y)

            if center <= P_x <= length - 5 and 4 <= P_y <= width - 5:
                break
        return P_x, P_y

    def generate_p_down_left(length, width, left_or_right):
        """Generate a random point P within the given range, using normal distribution"""
        center = length / 2

        if left_or_right == 'left':
            mean_x_left = center / 2 + 2 - 10
        elif left_or_right == 'right':
            mean_x_left = center / 2 + 2 + 10
        else:
            raise ValueError("left_or_right must be 'left' or 'right'")

        while True:
            P_x = random.gauss(mean_x_left, sigma_solution)

            # Generate P_y using gamma distribution and map to interval [0, length-24.357]
            shape_y = 3
            scale_y = 0.05 * width
            P_y_gamma = np.random.gamma(shape_y, scale_y)

            # map to [0, length-25.357]
            P_y = (width - 25.357) - P_y_gamma

            if 5 <= P_x <= length - 5 and 0 <= P_y <= width - 24.357:
                break

        return P_x, P_y

    def generate_p_down_right(length, width, left_or_right):
        """Generate a random point P within the given range, using normal distribution"""
        center = length / 2

        if left_or_right == 'left':
            mean_x_right = length - center / 2 - 2 - 10
        elif left_or_right == 'right':
            mean_x_right = length - center / 2 - 2 + 10
        else:
            raise ValueError("left_or_right must be 0 (left) or 1 (right)")

        while True:
            P_x = random.gauss(mean_x_right, sigma_solution)

            shape_y = 3
            scale_y = 0.05 * width
            P_y_gamma = np.random.gamma(shape_y, scale_y)

            P_y = (width - 24.357) - P_y_gamma

            if 5 <= P_x <= length - 5 and 0 <= P_y <= width - 24.357:
                break

        return P_x, P_y



    total_volume_wear = 0
    total_weared_unit = 0

    theta_value = math.cos(theta)

    # Create the sandbox (representing one step)
    sandbox = SandBox(length, width, height, mu_0, mu_min, theta, materials, k, F, H)

    # Display sandbox info
    sandbox.display_info()

    # Visualize the surface wear of the sandbox
    # sandbox.visualize_surface()

    all_elements = sandbox.all_elements
    surface_elements = sandbox.surface_elements


    # /* ========================= Monte Carlo Simulation ========================== */

    print(f"\n@@@@ Monte Carlo START, total epochs: {epoches} @@@@\n")

    flag_epoch = [0, None]


    for epoch in tqdm(range(epoches), desc="(UP) Monte Carlo Simulation", unit="Epochs"):
        random_float = random.random()

        if random_float >= 0.5:
            flag_feet = "left"
        else:
            flag_feet = "right"

        # print(f"this time is: {flag_feet}")
        # print(f"flag_epoch is: {flag_epoch}")

        if flag_epoch[0] == 1:  # We only did the left side by side last time. Now do the right side by side
            # print("into sideBySide Second, i.e., Right")
            flag_epoch[0] = 0

            # Use the last state bequeathed to flag_epoch to indicate whether the current state is UP or DOWN
            if flag_epoch[1] == 'up':
                random_float = random.random()
                if random_float < 0.5: flag_feet = "left"
                else: flag_feet = "right"
                P_x, P_y = generate_p_up_right(length, width, flag_feet)  # on the right side, Up
                wear_elements = get_wear_elements_up(surface_elements, P_x, P_y, r, c)
            elif flag_epoch[1] == 'down':
                random_float = random.random()
                if random_float < 0.5: flag_feet = "left"
                else: flag_feet = "right"
                P_x, P_y = generate_p_down_right(length, width, flag_feet)  # on the right side, Down
                wear_elements = get_wear_elements_down(surface_elements, P_x, P_y, r, c)
            else:
                print("Error: flag_epoch[1] is None")
                exit(1)

            flag_epoch[1] = None

        else:

            # /* ============= Mix: up and down the stairs, single side by side ratio (mix ratio) radio ========== */

            random_flag_single_sideBySide = random.random()
            random_flag_up_down = random.random()


            if random_flag_single_sideBySide > single_sideBS:  # If you exceed the threshold for side-by-side walking, you're walking alone
                single_or_sideBySide = 'single'
                # print("into single")

                # Currently 1 person. Start judging whether the person is going up or down stairs
                if random_flag_up_down < up_down_radio:
                    flag_up_down = 'up'
                    random_float = random.random()
                    if random_float < 0.5:
                        flag_feet = "left"
                    else:
                        flag_feet = "right"
                    P_x, P_y = generate_p_single_up(length, width, flag_feet)
                    wear_elements = get_wear_elements_up(surface_elements, P_x, P_y, r, c)
                else:
                    flag_up_down = 'down'
                    random_float = random.random()
                    if random_float < 0.5:
                        flag_feet = "left"
                    else:
                        flag_feet = "right"
                    P_x, P_y = generate_p_single_down(length, width, flag_feet)
                    wear_elements = get_wear_elements_down(surface_elements, P_x, P_y, r, c)

            # Two people are walking side by side, assuming the same direction
            else:
                # print("into sideBySide First, i.e., Left")
                single_or_sideBySide = 'sideBySide'

                flag_epoch[0] = 1
                # print("Successfully set flag_epoch[0] to 1")

                if random_flag_up_down < up_down_radio:
                    flag_epoch[1] = 'up'
                    random_float = random.random()
                    if random_float >= 0.5:
                        flag_feet = "left"
                    else:
                        flag_feet = "right"
                    P_x, P_y = generate_p_up_left(length, width, flag_feet)  # Walking on the left side
                    wear_elements = get_wear_elements_up(surface_elements, P_x, P_y, r, c)
                else:
                    flag_epoch[1] = 'down'
                    random_float = random.random()
                    if random_float >= 0.5:
                        flag_feet = "left"
                    else:
                        flag_feet = "right"
                    P_x, P_y = generate_p_down_left(length, width, flag_feet)  # Walking on the left side
                    wear_elements = get_wear_elements_down(surface_elements, P_x, P_y, r, c)


        # S_shoe = 26000  # in mm^3
        S_shoe = 260  # in cm^3
        h_mean = np.mean([element.h for element in wear_elements])
        mu_mean = calculate_mu_mean_speed(wear_elements)
        # s = calculate_sliding_displacement_speed(mu_mean, theta, v, g)
        s = calculate_sliding_displacement_speed_fast(mu_mean, theta_value, v, g)
        volume_wear = calculate_wear_volume_speed(k, F, s, H)

        total_volume_wear += volume_wear*1000
        sum_each_wear = 0
        this_epoch_wear_unit = 0

        # /* ======== Adaptively Calculate the Wear Weight Ratio ========= */
        # According to the depth h, adaptively calculate the wear weight ratio
        wear_reserved = 1
        h_max = max([element.h for element in wear_elements])
        total_weight = sum([(h_max + wear_reserved) - element.h for element in wear_elements])

        if silent == False:
            print(f"Surface elements Counter: {len(surface_elements)}")
            print(f"Total wear elements: {len(wear_elements)}")
            print(f"Shoe model in Stair percent: {round((len(wear_elements) / S_shoe), 4) * 100}%")
            print(f"h_mean: {h_mean} (cm)")
            print(f"mu_mean: {mu_mean}")
            print(f"epoch: {epoch}")
            print(f"s: {s} (mm)")
            print(f"V: {volume_wear} (mm^3)")
            print(f"h_max: {h_max} (cm)")
            print(f"total_weight: {total_weight}")

        # Apply wear to each element based on the weight ratio
        for element in wear_elements:
        # for element in tqdm(wear_elements):
            weight_ratio = ((h_max + wear_reserved) - element.h) / total_weight
            # wear_hp = weight_ratio * volume_wear * 1000  # Scale to match the HP value (since each unit is 1mm³)

            wear_hp = np.int32(weight_ratio * volume_wear)
            # since each hp is 1mm³
            # volume_wear in mm³


            if wear_hp < 0:
                element.show_elements()
                print(f"weight_ratio: {weight_ratio}")
                print(f"volume_wear: {volume_wear}")
                print(f"total_weight: {total_weight}")
            # print(f"element hp: {element.hp}, wear hp: {wear_hp}")

            if element.hp < wear_hp:  # If it is fully worn, add the next particle to the worn list
                sum_each_wear += element.hp*1000

                # Set the hp of the current particle to 0, which means full wear
                element.hp = 0

                rest_wear_hp = wear_hp - element.hp
                # Remove the current particle from the wear list and add the particle with h = h+1 (one layer deeper) to the wear list

                # Remove the current element from the wear_elements list
                wear_elements.remove(element)
                surface_elements.remove(element)
                total_weared_unit += 1
                this_epoch_wear_unit += 1

                # Find the next layer's (h+1) element and add it to the wear_elements list
                if element.h < height:  # Ensure that we don't go below the surface (h should not be negative)
                    new_element = sandbox.get_element(element.x, element.y, element.h + 1)

                    # # show Old and New info change
                    # print(f"### Wear one whole unit, in Epoch {epoch}, old element:")
                    # element.show_elements()
                    # print(f"### New element:")
                    # new_element.show_elements()

                    wear_elements.append(new_element)
                    surface_elements.append(new_element)
                    new_element.hp -= rest_wear_hp
                    sum_each_wear += rest_wear_hp * 1000

            else:  # /* element.hp >= wear_hp */
                element.hp -= wear_hp
                # Print the current status of the worn particles after each wear
                # element.show_elements()

                sum_each_wear += wear_hp * 1000

        if silent == False:
            print(f"This Epoch weared unit: {this_epoch_wear_unit}")
            print(f"Sum Each Wear: {sum_each_wear}")


    print(f"\n@@@@ Monte Carlo END @@@@")
    print(f"Total step: {total_step}")
    print(f"Total volume wear: {total_volume_wear}")
    print(f"Total weared unit: {total_weared_unit}")

    print("\n show Sandbox information: \n")
    sandbox.display_info()

    print(f"\nShow visualize surface from elements: \n")
    print(f"##  total Length of surface_elements: {len(surface_elements)}  ##")

    save_surface_elements_to_json(sandbox.surface_elements)

    print(f"\n  @@@@  total weared UNIT: {total_weared_unit}  @@@@  ")


    # /* ================= Visualize surface ================= */
    if visualize:
        # sandbox.visualize_surface()

        sandbox.visualize_surface_from_elements()

        visualize_surface_plus_layer(sandbox)

        visualize_surface_pro_max(sandbox, up_down_radio, single_sideBS)

        visualize_surface_pro_max_plotly(sandbox)

        visualize_surface_pro_max_surface(sandbox)


if __name__ == "__main__":
    main()


