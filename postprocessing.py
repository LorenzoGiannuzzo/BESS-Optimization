import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def total_convergence(n_gen, timewindow, pop_size, X, Y):
    # Convert Y to a DataFrame (if not already)
    Y = pd.DataFrame(Y)

    # Calculate statistics of Y
    Y_stats = Y.transpose().describe()
    Y_stats = Y_stats.transpose()
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a custom colormap
    cmap_colors = ["orange", "darkorchid", "indigo"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", cmap_colors)

    # Plot the mean from Y_stats with customization
    Y_stats['mean'].plot(ax=ax, color=cmap(0.5), linestyle='-', linewidth=2,
                         label='Mean Fitness Value of Population')  # Customize the plot with linestyle, linewidth, and label
    ax.legend()

    # Add titles and axis labels
    ax.set_title('Statistics of Fitness', fontsize=16)  # Title of the plot
    ax.set_xlabel('Categories', fontsize=14)  # X-axis label
    ax.set_ylabel('Values', fontsize=14)  # Y-axis label

    # Customize background and grid
    ax.set_facecolor('whitesmoke')  # Background color
    ax.grid(True, linestyle='--', linewidth=0.5, color='white')  # Dashed grid lines with 0.5 linewidth and gray color

    # Save the figure
    plt.savefig('Plots/total_convergence.png')

