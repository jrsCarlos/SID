import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def draw_history(history, title, filename=None, save_dir="plots"):
    data = pd.DataFrame({'Episode': range(1, len(history) + 1), title: history})
    
    # creem la imatge
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y=title, data=data)
    
    # Personalizar
    plt.title(f"{title} Over Episodes")
    plt.xlabel('Episode')
    plt.ylabel(title)
    plt.grid(True)
    plt.tight_layout()

    # Preparar nombre de archivo
    if filename is None:
        filename = f"{title.replace(' ', '_').lower()}.png"
    
    # Crear carpeta si no existe
    os.makedirs(save_dir, exist_ok=True)

    # Guardar figura
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Gráfica guardada en: {filepath}")
