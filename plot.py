import pandas as pd
import matplotlib.pyplot as plt

# cargar datos
df = pd.read_csv("docs/assets/matrix_results.csv")

# gráfica de tiempos
plt.figure()
plt.plot(df["n"], df["serial"], marker="o", label="Serial")
plt.plot(df["n"], df["rows"], marker="o", label="Rows")
plt.plot(df["n"], df["cols"], marker="o", label="Cols")
plt.plot(df["n"], df["blocks"], marker="o", label="Blocks")

plt.xlabel("Matrix Size (n)")
plt.ylabel("Time (seconds)")
plt.title("Matrix Multiplication Performance")
plt.legend()

plt.savefig("docs/assets/plots/matrix_times.png")
plt.show()

# gráfica de speedup
plt.figure()
plt.plot(df["n"], df["speedup_rows"], marker="o", label="Rows")
plt.plot(df["n"], df["speedup_cols"], marker="o", label="Cols")
plt.plot(df["n"], df["speedup_blocks"], marker="o", label="Blocks")

plt.xlabel("Matrix Size (n)")
plt.ylabel("Speedup")
plt.title("Speedup Comparison")
plt.legend()

plt.savefig("docs/assets/plots/matrix_speedup.png")
plt.show()