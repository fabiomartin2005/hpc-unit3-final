import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("docs/assets/image_results.csv")

plt.plot(df["workers"], df["parallel"], marker="o", label="Parallel")
plt.axhline(df["serial"][0], linestyle="--", label="Serial")

plt.xlabel("Workers")
plt.ylabel("Time (seconds)")
plt.title("Image Processing Performance")

plt.legend()

plt.savefig("docs/assets/plots/image_plot.png")
plt.show()