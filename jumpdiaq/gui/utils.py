# builtin
import re
# external
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def extract_unique_parts(strings):
    if not strings:
        return strings

    # Step 1: Split each string by '_' and '-'
    split_strings = [re.split(r'[-_]', s) for s in strings]

    # Step 2: Transpose the list to identify common elements **at the same position**
    transposed = list(zip(*split_strings))

    # Step 3: Identify indices where all elements are identical
    common_indices = {i for i, values in enumerate(transposed) if len(set(values)) == 1}

    # Step 4: Remove elements at common indices
    unique_parts = [
        [value for i, value in enumerate(parts) if i not in common_indices]
        for parts in split_strings
    ]

    # Step 5: Reconstruct the unique strings
    return ['_'.join(parts) for parts in unique_parts]


# Helper functions
def rmeans(data_array, axis=1):
    """Calculate row means, ignoring NaN values"""
    return np.nanmean(data_array, axis=axis)

def rcvs(data_array):
    """Calculate coefficient of variation for each row, ignoring NaN values"""
    return np.nanstd(data_array, axis=1) / np.nanmean(data_array, axis=1)

def rsums(data_array, axis=1):
    """Calculate row sums, treating NaN as 0"""
    return np.sum(~np.isnan(data_array), axis=axis)

def alpha(hex_color, alpha_value):
    """Add alpha transparency to hex color"""
    # Convert hex to rgba
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba{rgb + (alpha_value,)}'


def create_box_hist_plot(x, labels, colors, ylim=None, text_cex=1):
    df = pd.DataFrame({'Value': x, 'Label': labels})
    df = df[df['Value'].notna()]  

    fig = px.violin(
        df,
        y="Value",
        x="Label",
        color="Label",
        box=True,
        points=False,
        color_discrete_sequence=colors
    )

    fig.update_layout(
        font=dict(size=14 * text_cex),
        xaxis_title='',
        yaxis_title=None,  # Explicitly remove the y-axis title
        margin=dict(l=10),  # Reduce left margin (l = left)
        showlegend=False,
        violingap=0,
        violingroupgap=0,
    )

    if ylim is not None:
        fig.update_yaxes(range=ylim)
        
    # Add count annotations above each violin
    group_counts = df['Label'].value_counts()
    for label, count in group_counts.items():
        fig.add_annotation(
            x=label,
            y=df['Value'].min() * 0.95 if ylim is None else ylim[0] * 0.95,
            text=f"n={count}",
            showarrow=False,
            font=dict(size=12 * text_cex),
            xanchor='center'
        )

    return fig


def lfqbench_plot(mat, prot_tags, species_list, ratios, colors=['blue', 'chocolate'], 
                   cols_A=None, cols_B=None, span=None, point_cex=0.3, 
                   sample_groups=None, title=None, text_cex=1):
    """
    Main function to create the LFQBench plot with Plotly
    """
    if sample_groups is None:
        sample_groups = ['A', 'B']
    
    # Handle single species case
    if isinstance(species_list, str):
        species_list = [species_list]
    if isinstance(ratios, (int, float)):
        ratios = [ratios]
    if isinstance(colors, str):
        colors = [colors]
    
    # Convert matrix to numpy array if it's not already
    if isinstance(mat, pd.DataFrame):
        mat_np = mat.values
    else:
        mat_np = np.array(mat)
    
    # Determine which type each item belongs to
    type_vec = []
    for t in prot_tags:
        found = False
        for tag_item in species_list:
            if re.search(tag_item, t):
                type_vec.append(tag_item.lower())
                found = True
                break
        if not found:
            type_vec.append(None)
    
    # Create log2 versions of the data
    #mat_log2 = np.log2(mat_np)
    mat_log2 = mat_np
    
    # Calculate means for samples A and B
    A_means = []
    B_means = []
    for i in range(mat_log2.shape[0]):
        row_A = [mat_log2[i, j] for j in cols_A if not np.isnan(mat_log2[i, j])]
        row_B = [mat_log2[i, j] for j in cols_B if not np.isnan(mat_log2[i, j])]
        
        A_means.append(np.mean(row_A) if row_A else np.nan)
        B_means.append(np.mean(row_B) if row_B else np.nan)
    
    # Calculate log2 ratios
    R = [a - b for a, b in zip(A_means, B_means)]
    
    # Count valid ratios per species
    nums = []
    for species_idx, species in enumerate(species_list):
        species_indices = [i for i, t in enumerate(type_vec) if t == species.lower()]
        valid_ratios = [R[i] for i in species_indices if np.isfinite(R[i])]
        nums.append(len(valid_ratios))
        print(f"{species}: valid ratios with at least two replicates in each of A and B: {len(valid_ratios)}")
    
    # Determine y-axis range if not provided
    if span is None:
        valid_ratios = [r for r in R if np.isfinite(r)]
        if valid_ratios:
            span = max(abs(min(valid_ratios)), abs(max(valid_ratios))) * 1.1
        else:
            span = 2  # Default if no valid ratios
    
    # Create scatter plot
    scatter_fig = go.Figure()
    
    for species_idx, species in enumerate(species_list):
        species_indices = [i for i, t in enumerate(type_vec) if t == species.lower()]
        species_B = [B_means[i] for i in species_indices]
        species_R = [R[i] for i in species_indices]
        
        # Filter out NaN values
        valid_indices = [(B, ratio) for B, ratio in zip(species_B, species_R) 
                         if np.isfinite(B) and np.isfinite(ratio)]
        
        if valid_indices:
            species_B_valid, species_R_valid = zip(*valid_indices)
            
            # Adjust opacity based on number of points
            opacity = min(1.0, 15 / np.sqrt(len(species_B_valid)))
            
            scatter_fig.add_trace(go.Scatter(
                x=species_B_valid,
                y=species_R_valid,
                mode='markers',
                marker=dict(
                    color=colors[species_idx],
                    opacity=opacity,
                    size=point_cex*10
                ),
                name=species
            ))
    
    # Add horizontal lines at expected ratios
    x_values = [b for b in B_means if np.isfinite(b)]
    for i, ratio in enumerate(ratios):
        scatter_fig.add_shape(
            type="line",
            x0=min(x_values, default=0),
            y0=np.log2(ratio),
            x1=max(x_values, default=1),
            y1=np.log2(ratio),
            line=dict(color='rgba(255,0,0,0.5)', width=2, dash='dash')
        )
    
    # Update scatter plot layout
    scatter_fig.update_layout(
        xaxis_title=f'Log2 ({sample_groups[1]})',
        yaxis_title=f'Log2 ({sample_groups[0]} / {sample_groups[1]})',
        yaxis=dict(range=[-span, span]),
        xaxis=dict(
            range=[
                np.quantile(x_values, 0.01) if len(x_values) else 0,
                np.quantile(x_values, 0.999) if len(x_values) else 1
            ]
        ),
        showlegend=False,
        plot_bgcolor='white',
        title=title,
        font=dict(size=14*text_cex),
    )
    
    # Create box-histogram plot
    box_hist_fig = create_box_hist_plot(
        R, type_vec, 
        colors,
        ylim=[-span, span]
    )
    
    # Add horizontal lines at expected ratios
    for ratio in ratios:
        box_hist_fig.add_shape(
            type="line",
            x0=0,
            y0=np.log2(ratio),
            x1=3,
            y1=np.log2(ratio),
            line=dict(color='rgba(255,0,0,0.5)', width=2, dash='dash')
        )
    
    # Return all three plots
    return scatter_fig, box_hist_fig