import pandas as pd
from serial import run as run_serial
from parallel_rows import run as run_rows
from parallel_cols import run as run_cols
from parallel_blocks import run as run_blocks


def main():
    # ⚠️ tamaños pequeños para evitar que se congele
    sizes = [100, 150, 200]
    workers = 4

    results = []

    for n in sizes:
        print(f"\nRunning size {n}...")

        t_serial = run_serial(n)
        t_rows = run_rows(n, workers)
        t_cols = run_cols(n, workers)
        t_blocks = run_blocks(n, workers)

        results.append({
            "n": n,
            "serial": t_serial,
            "rows": t_rows,
            "cols": t_cols,
            "blocks": t_blocks,
            "speedup_rows": t_serial / t_rows,
            "speedup_cols": t_serial / t_cols,
            "speedup_blocks": t_serial / t_blocks
        })

    df = pd.DataFrame(results)

    print("\n=== RESULTS ===")
    print(df)

    # 📁 guarda resultados
    df.to_csv("docs/assets/matrix_results.csv", index=False)
    print("\nCSV saved in docs/assets/matrix_results.csv")


# 🔥 ESTO ES LO QUE ARREGLA EL ERROR EN WINDOWS
if __name__ == "__main__":
    main()