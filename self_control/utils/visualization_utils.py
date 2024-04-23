import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def draw_norms(concated_norms, tokens, attribute, top_k=50):
    # Grid dimensions
    rows, cols = concated_norms.shape[0], concated_norms.shape[1]

    # Generating random data for demonstration
    # Replace this with your actual data
    data = concated_norms

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create a colormap: white to red
    cmap = plt.cm.Reds
    cmap.set_under(color='white')

    # Plotting the heatmap
    cax = ax.matshow(data, cmap=cmap, aspect='auto', vmin=0.0001)

    # Optional: Add a color bar
    fig.colorbar(cax)

    # Setting the ticks for y-axis and x-axis
    ax.set_yticks(np.arange(rows))
    ax.set_xticks(np.arange(cols))  # Adjust step as needed

    # Labels for the ticks
    ax.set_yticklabels(np.arange(rows))
    ax.set_xticklabels(tokens, rotation=-60)  # Adjust step as needed

    indices = np.unravel_index(np.argsort(concated_norms.ravel())[-top_k:], concated_norms.shape)

    # Overlay rectangles for top-50 norms
    for y, x in zip(*indices):
        # Add a rectangle with bold borders
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    ax.set_ylabel('Layer ID')
    ax.set_xlabel('Token Index')

    # Adding a caption
    caption = f"Norms of each gradients. Attribute: {attribute}"  # Replace with your actual caption
    fig.text(0.45, 1.05, caption, ha='center', va='top', fontsize=12, color='black')

    # Show the plot
    plt.show()
