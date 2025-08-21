# builtin
import re
import socket
# external
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_local_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"  # Fallback to localhost

def ratio_violin_plot(ratios, labels, colors, ylim=None, text_cex=1):
    df = pd.DataFrame({'Value': ratios, 'Label': labels})
    df = df[df['Value'].notna()]  

    fig = px.violin(
        df,
        y="Value",
        x="Label",
        color="Label",
        box=False,
        points=False,
        color_discrete_map=colors
    )

    fig.update_layout(
        font=dict(size=14 * text_cex),
        xaxis_title='Species',
        yaxis_title=None,  # Explicitly remove the y-axis title
        margin=dict(l=10),  # Reduce left margin (l = left)
        showlegend=False,
        violingap=0,
        violingroupgap=0,
    )
    
    fig.update_traces(
        box=dict(visible=False),
        meanline=dict(visible=False),
        scalemode='width',
        spanmode='hard',
    )
    
    if ylim is not None:
        fig.update_yaxes(range=ylim)
        
    # Calculate statistics for each group
    group_stats = df.groupby('Label')['Value'].agg(['count', 'median', 'mean', 'std'])
    
    # Add annotations for each violin
    for label in group_stats.index:
        count = group_stats.loc[label, 'count']
        median = group_stats.loc[label, 'median']
        mean = group_stats.loc[label, 'mean']
        std = group_stats.loc[label, 'std']
        
        # Position for count annotation (below violin)
        y_count = df['Value'].min() * 0.95 if ylim is None else ylim[0] * 0.95
        
        # Position for statistics annotation (above violin)
        y_stats = df['Value'].max() * 0.85 if ylim is None else ylim[1] * 0.85
        
        # Add count annotation
        fig.add_annotation(
            x=label,
            y=y_count,
            text=f"n={count}",
            showarrow=False,
            font=dict(size=12 * text_cex),
            xanchor='center',
            bgcolor="rgba(255,255,255,0.5)",
        )
        
        # Add statistics annotation
        stats_text = f"Median: {median:.2f}<br>Mean: {mean:.2f}<br>SD: {std:.2f}"
        fig.add_annotation(
            x=label,
            y=y_stats,
            text=stats_text,
            showarrow=False,
            font=dict(size=10 * text_cex),
            xanchor='center',
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="gray",
            borderwidth=0.2
        )

    return fig


def get_valid_ratios(mat, prot_tags, prot_species, species_list, sample_groups, cols_A, cols_B):
    if sample_groups is None:
        sample_groups = ['A', 'B']
    
    # Handle single species case
    if isinstance(species_list, str):
        species_list = [species_list]
    
    # Convert matrix to numpy array if it's not already
    if isinstance(mat, pd.DataFrame):
        mat_np = mat.values
    else:
        mat_np = np.array(mat)
    
    # Create log2 versions of the data
    mat_log2 = mat_np
    # Calculate means for samples A and B
    A_means = []
    B_means = []
    for i in range(mat_log2.shape[0]):
        values_A = [mat_log2[i, j] for j in cols_A if not np.isnan(mat_log2[i, j])]
        values_B = [mat_log2[i, j] for j in cols_B if not np.isnan(mat_log2[i, j])]
        # only compute the mean when there are more than two values in each group
        A_means.append(np.mean(values_A) if len(values_A) > 1 else np.nan)
        B_means.append(np.mean(values_B) if len(values_B) > 1 else np.nan)
    
    # Calculate log2 ratios
    R = [a - b for a, b in zip(A_means, B_means)]       
    
    # Determine which type each item belongs to
    species_vec = []
    for s in prot_species:
        found = False
        for species in species_list:
            if re.search(species, s):
                species_vec.append(species)
                found = True
                break
        if not found:
            species_vec.append(None)
    
    valid_prots = {}
    for species in species_list:
        species_indices = [i for i, s in enumerate(species_vec) if s == species]
        species_prot_tag = [prot_tags[i] for i in species_indices]
        species_B = [B_means[i] for i in species_indices]
        species_R = [R[i] for i in species_indices]
        
        # Filter out NaN values
        valid_xys = [(prot_tag, B, ratio) for prot_tag, B, ratio in zip(species_prot_tag, species_B, species_R) 
                     if np.isfinite(B) and np.isfinite(ratio)]
        valid_prots[species] = valid_xys
        #print(f"{species}: valid ratios with at least two replicates in each of A and B: {len(valid_xys)}")
    
    return valid_prots


def lfqbench_plot(valid_prots, species_list, exp_ratios, colors, 
                  span=4, point_cex=0.3, text_cex=1, opacity=0.8, 
                  sample_groups=None, title=None):
    """
    Main function to create the LFQBench plot with Plotly
    """     
    
    # Create scatter plot
    ratio_intensity_fig = go.Figure()
    
    all_valid_prot_tags = []
    all_valid_B = []
    all_valid_R = []
    species_vec = []
    for species in species_list:
        valid_xys = valid_prots[species]
        if valid_xys:
            prot_tag_valid, species_B_valid, species_R_valid = zip(*valid_xys)
            all_valid_prot_tags = all_valid_prot_tags + list(prot_tag_valid)
            all_valid_B = all_valid_B + list(species_B_valid)
            all_valid_R = all_valid_R + list(species_R_valid)
            species_vec = species_vec + [species] * len(species_B_valid)
            
            ratio_intensity_fig.add_trace(go.Scatter(
                x=species_B_valid,
                y=species_R_valid,
                mode='markers',
                marker=dict(
                    color=colors[species],
                    opacity=opacity,
                    size=point_cex*10
                ),
                name=species,
                text=prot_tag_valid,
            ))
    
    for ratio in exp_ratios:
        # Add horizontal lines at expected ratios
        ratio_intensity_fig.add_shape(
            type="line",
            x0=min(all_valid_B, default=0),
            y0=np.log2(ratio),
            x1=max(all_valid_B, default=1),
            y1=np.log2(ratio),
            line=dict(color='rgba(0,0,0,0.5)', width=2, dash='dash')
        )
    
    # Update scatter plot layout
    if sample_groups is None:
        sample_groups = ['A', 'B']
    ratio_intensity_fig.update_layout(
        xaxis_title=f'Log2 ({sample_groups[1]})',
        yaxis_title=f'Log2 ({sample_groups[0]} / {sample_groups[1]})',
        yaxis=dict(range=[-span, span]),
        xaxis=dict(
            range=[
                np.quantile(all_valid_B, 0.01) if len(all_valid_B) else 0,
                np.quantile(all_valid_B, 0.999) if len(all_valid_B) else 1
            ]
        ),
        showlegend=False,
        plot_bgcolor='white',
        title=title,
        font=dict(size=14*text_cex),
    )
    
    # Create box-histogram plot
    ratio_dist_fig = ratio_violin_plot(
        all_valid_R, 
        species_vec, 
        colors,
        ylim=[-span, span]
    )
    
    # Add horizontal lines at expected ratios
    for ratio in exp_ratios:
        ratio_dist_fig.add_shape(
            type="line",
            x0=-1,
            y0=np.log2(ratio),
            x1=3,
            y1=np.log2(ratio),
            line=dict(color='rgba(0,0,0,0.5)', width=2, dash='dash')
        )
    
    return ratio_intensity_fig, ratio_dist_fig

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