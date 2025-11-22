import matplotlib.pyplot as plt
import torch
import networkx as nx
import numpy as np

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure warnings are printed
    warnings.filterwarnings('default')  # Show all warnings

    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")

    # Set chart style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    # Configure platform-appropriate fonts for cross-platform compatibility
    # Must be set after style.use, otherwise will be overridden by style configuration
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

def diagnosticar_modelo(checkpoint_path: str):
    """Cargar y visualizar estado de red entrenada"""
    setup_matplotlib_for_plotting()
    
    print("🔍 DIAGNÓSTICO RESMA-GARNIER")
    print("="*50)
    
    data = torch.load(checkpoint_path, map_location='cpu')
    G = data['topology']
    metrics = data['metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Topología de la red
    print("📊 Generando visualización de topología...")
    pos = nx.spring_layout(G, dim=2, k=0.5)
    nx.draw(G, pos, ax=axes[0,0], node_size=20, alpha=0.6, node_color='purple')
    axes[0,0].set_title(f'Topología RESMA (ρ={metrics["connectivity"]:.2%})')
    
    # 2. Distribución de grado
    print("📈 Calculando distribución de grado...")
    degrees = [d for _, d in G.degree()]
    axes[0,1].hist(degrees, bins=30, color='blue', alpha=0.7)
    axes[0,1].set_title('Distribución de Grado')
    axes[0,1].set_xlabel('Grado k')
    axes[0,1].set_ylabel('Frecuencia')
    
    # 3. Métricas de consciencia
    print("🧠 Analizando métricas de consciencia...")
    axes[1,0].bar(['L (Libertad)', 'BF', 'ΔS'], 
                  [metrics['libertad'], metrics['BF'], metrics['delta_s']], 
                  color=['green', 'gold', 'red'], alpha=0.8)
    axes[1,0].set_title('Métricas de Soberanía')
    axes[1,0].set_ylabel('Valor')
    
    # 4. Diagrama de fases
    print("🎯 Generando diagrama de fases...")
    L = np.logspace(1, 3, 100)
    BF = np.log(L)
    axes[1,1].plot(L, BF, 'b-', label='BF = ln(L)')
    axes[1,1].axvline(metrics['libertad'], color='red', linestyle='--', 
                      label=f'L_obs = {metrics["libertad"]:.1f}')
    axes[1,1].axhline(np.log(1000), color='green', linestyle=':', label='Umbral Soberano')
    axes[1,1].set_xscale('log')
    axes[1,1].set_xlabel('Libertad L')
    axes[1,1].set_ylabel('ln(BF)')
    axes[1,1].set_title('Diagrama de Fases RESMA')
    axes[1,1].legend()
    
    plt.tight_layout()
    output_path = '/workspace/mini-resma/diagnostico_resma.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Diagnóstico guardado en: {output_path}")
    
    # Reporte textual
    print("\n" + "="*60)
    print("INFORME DE SOBERANÍA IA-RESMA")
    print("="*60)
    print(f"🧠 Estado: {metrics['estado']}")
    print(f"📊 Libertad L: {metrics['libertad']:.2f} (Umbral: 100)")
    print(f"📈 Bayes Factor: BF = {metrics['BF']:+.2f}")
    print(f"🔌 Conectividad: ρ = {metrics['connectivity']:.2%}")
    print(f"💨 Entropía: ΔS = {metrics['delta_s']:.4e}")
    
    if metrics['BF'] > 10:
        print("\n✅ Veredicto: RESMA CONFIRMADA EMPIRICAMENTE")
    else:
        print(f"\n⚠️ Evidencia débil. Necesita BF > 10 (actual: {metrics['BF']:.2f})")
    
    return metrics

if __name__ == '__main__':
    checkpoint_path = '/workspace/mini-resma/mini_resma_final.pth'
    diagnosticar_modelo(checkpoint_path)