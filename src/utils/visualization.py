"""
MolGuard — Görselleştirme Yardımcıları / Visualization Helpers
Molekül, grafik ve sonuç görselleştirmeleri
Molecule, graph, and result visualizations
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ATOM_MAP = {
    6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P',
    16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

COLOR_MAP = {
    'C': '#404040', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'S': '#FFFF30', 'P': '#FF8000',
    'Cl': '#1FF01F', 'Br': '#A62929', 'I': '#940094'
}


def mol_to_networkx(data):
    """PyG verisini NetworkX grafına dönüştür / Convert PyG to NetworkX"""
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()

    for n in range(data.x.shape[0]):
        atom_num = data.x[n][0].item()
        G.add_node(n, label=ATOM_MAP.get(atom_num, '?'))

    for e in range(edge_index.shape[1]):
        s, t = edge_index[0][e], edge_index[1][e]
        if s < t:
            G.add_edge(s, t)
    return G


def plot_molecule(data, title='Molekül / Molecule', ax=None):
    """Molekülü görselleştir / Visualize molecule"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    G = mol_to_networkx(data)
    pos = nx.spring_layout(G, seed=42, k=2)
    labels = nx.get_node_attributes(G, 'label')
    colors = [COLOR_MAP.get(labels[n], '#999') for n in G.nodes()]

    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels,
            node_color=colors, node_size=500, font_size=11,
            font_weight='bold', font_color='white',
            edge_color='#CCC', width=2)
    ax.set_title(title, fontweight='bold')
    return ax


def plot_importance(data, importance, title='Önem / Importance', ax=None):
    """Düğüm önem haritası / Node importance heatmap"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    G = mol_to_networkx(data)
    pos = nx.spring_layout(G, seed=42, k=2)
    labels = nx.get_node_attributes(G, 'label')

    cmap = plt.cm.YlOrRd
    node_colors = [cmap(importance[n]) for n in G.nodes()]
    node_sizes = [300 + 700 * importance[n] for n in G.nodes()]

    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels,
            node_color=node_colors, node_size=node_sizes,
            font_size=11, font_weight='bold', font_color='black',
            edge_color='#CCC', width=2)

    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Önem / Importance', shrink=0.8)
    ax.set_title(title, fontweight='bold')
    return ax
