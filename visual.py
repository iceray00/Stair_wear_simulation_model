import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import plotly.graph_objects as go



def visualize_surface_pro_max(Sandbox, caption_left='20', caption_right='50', name='3d_surface_pro_max'):
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

    # 设置坐标轴范围（核心修改点1：匹配输入尺寸）
    ax.set_xlim(0, Sandbox.length - 1)  # X轴范围按沙盒长度
    ax.set_ylim(0, Sandbox.width - 1)  # Y轴范围按沙盒宽度
    ax.set_zlim(0, Sandbox.height - 1)  # Z轴范围按沙盒高度

    # 设置三维盒状比例（核心修改点2：防止形变）
    ax.set_box_aspect([
        Sandbox.length,  # X轴比例因子 = 长度
        Sandbox.width,  # Y轴比例因子 = 宽度
        Sandbox.height  # Z轴比例因子 = 高度
    ])

    # 自定义渐变颜色
    colors = ["red", "yellow", "green", "blue"]
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_yellow_red", colors)

    # Plot the surface using mu as color mapping

    # cmap = 'cividis_r'
    scatter = ax.scatter(x_vals, y_vals, h_vals, c=mu_vals, cmap=cmap, s=1, alpha=1, norm = Normalize(vmin=Sandbox.mu_min, vmax=Sandbox.mu_0))

    ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Depth (H)')  # Use H as the depth, which is the negative direction
    ax.set_title(str(int(caption_left*100)) + " : " + str(int(caption_right*100)) + "", fontsize=18, pad=1)

    # Add color bar to indicate the friction coefficient (mu)
    # fig.colorbar(scatter, ax=ax, label='Friction Coefficient (mu)/Height (H)')

    # Invert the Z-axis to make depth go from top to bottom
    ax.invert_zaxis()

    name = str(int(caption_left*100)) + '_' + str(int(caption_right*100))
    result_file = 'results'
    if not os.path.exists(result_file):
        os.makedirs(result_file)
    plt.savefig(f'{result_file}/{name}.pdf')
    print(f"Save File in `results/{name}.pdf` Successfully!")
    plt.show()



def visualize_surface_pro_max_plotly(Sandbox):
    # Prepare data
    x_vals = []
    y_vals = []
    h_vals = []
    mu_vals = []

    for x in range(Sandbox.length):
        for y in range(Sandbox.width):
            for h in range(Sandbox.height):
                if x == 0 or x == Sandbox.length - 1 or y == 0 or y == Sandbox.width - 1 or h == 0 or h == Sandbox.height - 1:
                    element = Sandbox.get_element(x, y, h)
                    if element.hp > 0:
                        x_vals.append(x)
                        y_vals.append(y)
                        h_vals.append(h)
                        mu_vals.append(element.mu)

    for element in Sandbox.surface_elements:
        x_vals.append(element.x)
        y_vals.append(element.y)
        h_vals.append(element.h)
        mu_vals.append(element.mu)

    # Custom colorscale
    custom_colorscale = [
        [0, 'red'],
        [0.25, 'orange'],
        [0.5, 'yellow'],
        [0.75, 'green'],
        [1, 'blue']
    ]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=h_vals,
        mode='markers',
        marker=dict(
            size=4,  # Increase size for visibility
            color=mu_vals,  # Color based on mu
            colorscale=custom_colorscale,  # Use custom colorscale
            colorbar=dict(title="Friction Coefficient (mu)")
        )
    )])

    # Update layout for better visual appearance and proportional axes
    fig.update_layout(
        title="Visualization of Surface Wear",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Depth (H)',
            xaxis=dict(
                range=[0, Sandbox.length - 1]  # Set X axis range
            ),
            yaxis=dict(
                range=[0, Sandbox.width - 1]  # Set Y axis range
            ),
            zaxis=dict(
                range=[0, Sandbox.height - 1],  # Set Z axis range
                autorange='reversed'  # Reverse Z-axis
            ),
            aspectmode='manual',  # Manual aspect mode for control
            aspectratio=dict(
                x=Sandbox.length,  # Set X axis ratio to length
                y=Sandbox.width,  # Set Y axis ratio to width
                z=Sandbox.height  # Set Z axis ratio to height
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()



def visualize_surface_pro_max_surface(Sandbox):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data
    x_vals = []
    y_vals = []
    h_vals = []
    mu_vals = []

    for x in range(Sandbox.length):
        for y in range(Sandbox.width):
            for h in range(Sandbox.height):
                if x == 0 or x == Sandbox.length - 1 or y == 0 or y == Sandbox.width - 1 or h == 0 or h == Sandbox.height - 1:
                    element = Sandbox.get_element(x, y, h)
                    if element.hp > 0:
                        x_vals.append(x)
                        y_vals.append(y)
                        h_vals.append(h)
                        mu_vals.append(element.mu)

    for element in Sandbox.surface_elements:
        x_vals.append(element.x)
        y_vals.append(element.y)
        h_vals.append(element.h)
        mu_vals.append(element.mu)

    # Convert to numpy arrays for easier manipulation
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    h_vals = np.array(h_vals)
    mu_vals = np.array(mu_vals)

    # Create grid for surface
    X, Y = np.meshgrid(np.unique(x_vals), np.unique(y_vals))
    Z = np.zeros(X.shape)
    colors = np.zeros(X.shape + (4,))  # RGBA values

    # Assign values to the grid (interpolation for smooth surface)
    for i in range(len(x_vals)):
        ix = np.where(np.unique(x_vals) == x_vals[i])[0][0]
        iy = np.where(np.unique(y_vals) == y_vals[i])[0][0]
        Z[iy, ix] = h_vals[i]
        colors[iy, ix] = plt.cm.viridis(mu_vals[i])

    # 设置坐标轴范围（核心修改点1：匹配输入尺寸）
    ax.set_xlim(0, Sandbox.length - 1)  # X轴范围按沙盒长度
    ax.set_ylim(0, Sandbox.width - 1)  # Y轴范围按沙盒宽度
    ax.set_zlim(0, Sandbox.height - 1)  # Z轴范围按沙盒高度

    # 设置三维盒状比例（核心修改点2：防止形变）
    ax.set_box_aspect([
        Sandbox.length,  # X轴比例因子 = 长度
        Sandbox.width,  # Y轴比例因子 = 宽度
        Sandbox.height  # Z轴比例因子 = 高度
    ])

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, alpha=1, linewidth=0, antialiased=True)

    # Add color bar
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Friction Coefficient (mu)')

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Depth (H)')
    # ax.set_title('Surface Wear Visualization')
    ax.invert_zaxis()

    plt.savefig('max_surface.pdf')
    plt.show()
