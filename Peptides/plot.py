import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize

# ─── 0) Load your MATLAB‐style rcParams ─────────────────────────────────
mpl.rc_file('matlab.sty')
plt.style.use('matlab.sty')



# ─── 1)  Build “centre + rings” graph with sparse radial spokes ─────────
radii           = [1.5, 3.0, 4.5]
nodes_per_ring  = [6, 12, 18]
skip_radial     = 2            # only every 2nd node gets a spoke inward

G          = nx.Graph()
centre     = 'u'
G.add_node(centre)
positions  = {centre: (0.0, 0.0)}

for r, count in zip(radii, nodes_per_ring):
    for i in range(count):
        v = f"r{r:.1f}_n{i}"
        θ = 2 * np.pi * i / count
        positions[v] = (r * np.cos(θ), r * np.sin(θ))
        G.add_node(v)

        # tangential ring-edge
        prev = f"r{r:.1f}_n{(i - 1) % count}"
        G.add_edge(v, prev)

        # sparse radial spoke
        if i % skip_radial == 0:
            if r == radii[0]:                 # connect to centre
                G.add_edge(v, centre)
            else:                             # connect to inner ring
                inner_r     = radii[radii.index(r) - 1]
                inner_cnt   = nodes_per_ring[radii.index(r) - 1]
                j           = int(i * inner_cnt / count)
                G.add_edge(v, f"r{inner_r:.1f}_n{j}")

# collapse any multi-edges that might have sneaked in
G = nx.Graph(G)

# ─── 2)  Numerical helpers ------------------------------------------------
node_list   = list(G.nodes())
node_to_idx = {n: i for i, n in enumerate(node_list)}
N           = len(node_list)

# adjacency + symmetric‑normalised transition (for ChebNet step)
A     = nx.to_numpy_array(G, nodelist=node_list)
deg   = np.array([G.degree(n) for n in node_list])
D_inv_sqrt = np.diag(1 / np.sqrt(deg))
P     = D_inv_sqrt @ A @ D_inv_sqrt            # symmetric‑normalised

# radial distances (for the synthetic Euler‑style fields)
distances = np.array([np.hypot(*positions[n]) for n in node_list])
max_dist  = distances.max()

# ─── 3)  State arrays -----------------------------------------------------
# 3a)  Hot core + neighbours – common initial condition
T0 = np.zeros(N)
T0[node_to_idx[centre]] = 1.0
for nbr in G.neighbors(centre):
    T0[node_to_idx[nbr]] = 1.0

# 3b)  ChebNet diffusion sequence (top row)
T_cheb1 = P @ T0
T_cheb2 = P @ T_cheb1
cheb_states = [T0, T_euler1, T_cheb2]

# 3c)  Euler‑ChebNet synthetic diffusion fields (bottom row, middle + right)
#      Re‑using the functional forms from code2.py
decay        = 1.0 / max(radii)                # matches code2
T_baseline   = np.exp(-decay * distances)      # “Δt = 3” panel
T_euler1     = np.clip(1.0 - 0.5 * (distances / max_dist), 0, 1)  # “Δt = 2”
# left panel (bottom row) re‑uses the same T0 as initial condition
eul_states   = [T0, T_euler1, T_baseline]

# ─── 4)  Visual set‑up ----------------------------------------------------
cmap_nodes = LinearSegmentedColormap.from_list('YlRd_dark',
                                               ['#fdd835', '#b71c1c'])
norm_nodes = Normalize(vmin=0, vmax=1)
cmap_rings = ['#e8f5e9', '#d0f0c0', '#b2dfdb']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('white')

# white backdrop container across the whole grid (for cleaner edges)
bb0, bb_last = axes[0, 0].get_position(), axes[1, 2].get_position()
pad = 0.02
fig.add_artist(Rectangle((bb0.x0 - pad, bb_last.y0 - pad),
                         (bb_last.x1 - bb0.x0) + 2 * pad,
                         (bb0.y1 - bb_last.y0) + 2 * pad,
                         transform=fig.transFigure,
                         facecolor='white', edgecolor='none', zorder=-1))

# titles exactly as requested
row1_titles = ["ChebNet\nInitial", "Step=1", "Step=2"]
row2_titles = [
               "Euler-ChebNet \nInitial", "Step=1", "Step=2"]

# ─── 5)  Draw panels ------------------------------------------------------
for col, (title, T) in enumerate(zip(row1_titles, cheb_states)):
    ax = axes[0, col]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=8)

    # pastel radial rings
    for face, r in zip(cmap_rings, radii):
        ax.add_patch(Circle((0, 0), r, facecolor=face,
                            edgecolor='none', alpha=0.5, zorder=0))

    # edges
    nx.draw_networkx_edges(G, positions, ax=ax, alpha=0.3)

    if col == 0:   # initial: red core, grey elsewhere
        colors = ['#b71c1c' if T[node_to_idx[n]] == 1 else 'grey'
                  for n in node_list]
        nx.draw_networkx_nodes(G, positions, ax=ax,
                               node_color=colors,
                               edgecolors='black',
                               linewidths=1.0,
                               node_size=300)
    else:
        nx.draw_networkx_nodes(
                G,
                positions,
                ax=ax,
                node_color=T,          # array of scalar values
                cmap=cmap_nodes,
                vmin=0, vmax=1,        # << replace “norm=norm_nodes” with these
                edgecolors='black',
                linewidths=1.0,
                node_size=300
        )

for col, (title, T) in enumerate(zip(row2_titles, eul_states)):
    ax = axes[1, col]
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=8)

    # pastel radial rings
    for face, r in zip(cmap_rings, radii):
        ax.add_patch(Circle((0, 0), r, facecolor=face,
                            edgecolor='none', alpha=0.5, zorder=0))

    # edges
    nx.draw_networkx_edges(G, positions, ax=ax, alpha=0.3)

    if col == 0:   # initial panel (use same grey/red scheme)
        colors = ['#b71c1c' if T[node_to_idx[n]] == 1 else 'grey'
                  for n in node_list]
        nx.draw_networkx_nodes(G, positions, ax=ax,
                               node_color=colors,
                               edgecolors='black',
                               linewidths=1.0,
                               node_size=300)
    else:
        nx.draw_networkx_nodes(
                G,
                positions,
                ax=ax,
                node_color=T,          # array of scalar values
                cmap=cmap_nodes,
                vmin=0, vmax=1,        # << replace “norm=norm_nodes” with these
                edgecolors='black',
                linewidths=1.0,
                node_size=300
)


# ─── 6)  Shared colour‑bar (common scale of img2) ------------------------
sm   = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
sm.set_array([])
cax  = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Temperature', fontsize=14)

plt.subplots_adjust(left=0.05, right=0.9, wspace=0.15, hspace=0.25,
                    top=0.92, bottom=0.05)
plt.savefig('chebnet_rings.pdf', bbox_inches='tight')
plt.show()
### save as pdf

plt.close()
