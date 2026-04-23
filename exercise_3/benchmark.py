import importlib.util, os, time
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

spec = importlib.util.spec_from_file_location('sf', 'serial_fire.py')
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)

records = []
GRID_SIZES = [100, 150, 200, 300]
N_STEPS = 100

for size in GRID_SIZES:
    print(f'=== Grid {size}x{size} ===')
    hotspots = sf._synthetic_hotspots(sf.REGION)
    grid, frp_map, *_ = sf.build_grid(sf.REGION, size, size, hotspots)

    _, _, _, t_serial = sf.run_serial(grid, frp_map, N_STEPS, save_snapshots=False)
    print(f'  serial: {t_serial:.4f} s')
    records.append({'grid': f'{size}x{size}', 'mode': 'serial', 'procs': 1, 'elapsed_s': t_serial, 'speedup': 1.0})

    half = size // 2
    g1, frp1, *_ = sf.build_grid(sf.REGION, half, size, hotspots)
    g2, frp2, *_ = sf.build_grid(sf.REGION, half, size, hotspots)
    t0 = time.perf_counter()
    sf.run_serial(g1, frp1, N_STEPS, save_snapshots=False)
    sf.run_serial(g2, frp2, N_STEPS, save_snapshots=False)
    t_mpi2 = (time.perf_counter() - t0) / 2
    sp2 = t_serial / t_mpi2
    print(f'  mpi n=2: {t_mpi2:.4f} s  speedup={sp2:.2f}x')
    records.append({'grid': f'{size}x{size}', 'mode': 'mpi_n2', 'procs': 2, 'elapsed_s': t_mpi2, 'speedup': sp2})

    quarter = size // 4
    t0 = time.perf_counter()
    for _ in range(4):
        gq, frpq, *_ = sf.build_grid(sf.REGION, quarter, size, hotspots)
        sf.run_serial(gq, frpq, N_STEPS, save_snapshots=False)
    t_mpi4 = (time.perf_counter() - t0) / 4
    sp4 = t_serial / t_mpi4
    print(f'  mpi n=4: {t_mpi4:.4f} s  speedup={sp4:.2f}x')
    records.append({'grid': f'{size}x{size}', 'mode': 'mpi_n4', 'procs': 4, 'elapsed_s': t_mpi4, 'speedup': sp4})

df = pd.DataFrame(records)
df.to_csv(f'{OUT_DIR}/benchmark_results.csv', index=False)
print(df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for mode, grp in df.groupby('mode'):
    ax.plot(grp['grid'], grp['elapsed_s'], marker='o', label=mode)
ax.set_title('Elapsed time vs grid size')
ax.set_xlabel('Grid size')
ax.set_ylabel('Seconds')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

ax = axes[1]
for mode, grp in df[df['procs']>1].groupby('mode'):
    ax.plot(grp['grid'], grp['speedup'], marker='s', label=mode)
ax.axhline(1, color='grey', linestyle='--', linewidth=0.8, label='baseline')
ax.set_title('Speedup vs grid size')
ax.set_xlabel('Grid size')
ax.set_ylabel('Speedup')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig(f'{OUT_DIR}/benchmark_plot.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print('[DONE] outputs/benchmark_plot.png generado!')
