import matplotlib.pyplot as plt
import numpy as np

def visualize_uased_geometry():
    # Parámetros Físicos (RESMA Spec)
    energy_kev = 200.0
    lambda_pm = 1240.0 / (energy_kev * 1000 + energy_kev**2 / 511) # Relativista aprox (pm)
    # Lambda para 200keV es aprox 2.5 pm (0.025 Angstrom)
    lambda_angstrom = 0.0251 
    
    q0_target = 9.24  # Angstrom^-1 (Pico E8)
    
    # Ley de Bragg: q = 4π/λ * sin(θ)  => sin(θ) = q*λ / 4π
    # Ángulo de dispersión total (2θ) ≈ q*λ / 2π (para ángulos pequeños)
    theta_rad = np.arcsin(q0_target * lambda_angstrom / (4 * np.pi))
    scattering_angle_mrad = 2 * theta_rad * 1000
    
    # Configuración del Microscopio
    camera_length_mm = 1500.0 # Longitud de cámara efectiva (Modo Difracción)
    detector_size_mm = 40.0   # Detector directo estándar
    
    # Radio del anillo en el detector: R = L * tan(2θ)
    radius_mm = camera_length_mm * np.tan(2 * theta_rad)
    
    # Generar gráfico
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f0f0f0')
    
    # Dibujar Detector
    detector = plt.Rectangle((-detector_size_mm/2, -detector_size_mm/2), 
                             detector_size_mm, detector_size_mm, 
                             fill=True, color='#333333', alpha=0.1, label='Detector (4x4 cm)')
    ax.add_patch(detector)
    
    # Dibujar Anillo de Difracción E8
    circle = plt.Circle((0, 0), radius_mm, color='#d62728', fill=False, 
                        linestyle='--', linewidth=2, label=f'Anillo E8 ($q_0={q0_target} \AA^{{-1}}$)')
    ax.add_patch(circle)
    
    # Dibujar Haz Directo (Bloqueado)
    beamstop = plt.Circle((0, 0), 2.0, color='black', alpha=0.8, label='Beamstop')
    ax.add_patch(beamstop)
    
    # Calcular visibilidad
    visible = radius_mm < (detector_size_mm / 2)
    
    # Estética
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.set_xlabel('Posición en Detector X (mm)')
    ax.set_ylabel('Posición en Detector Y (mm)')
    ax.set_title(f'Geometría de Difracción UASED (E=200keV, L={camera_length_mm}mm)\n'
                 f'Posición del Pico RESMA: R = {radius_mm:.2f} mm')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    # Anotaciones
    status_color = 'green' if visible else 'red'
    status_text = "DETECTABLE" if visible else "FUERA DE RANGO"
    
    plt.text(0, -35, 
             f"Ángulo de Scattering: {scattering_angle_mrad:.2f} mrad\n"
             f"Estado: {status_text} (Necesita L ajustado)", 
             ha='center', fontsize=12, color=status_color, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor=status_color))

    plt.tight_layout()
    plt.savefig('uased_geometry.png')

visualize_uased_geometry()