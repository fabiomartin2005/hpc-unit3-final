import pandas as pd
from serial_pipeline import run as run_serial
from parallel_pipeline import run as run_parallel

def main():
    workers = [2, 4, 6]

    results = []

    t_serial = run_serial()

    for w in workers:
        t_parallel = run_parallel(w)

        results.append({
            "workers": w,
            "serial": t_serial,
            "parallel": t_parallel,
            "speedup": t_serial / t_parallel
        })

    df = pd.DataFrame(results)

    print(df)

    df.to_csv("docs/assets/image_results.csv", index=False)


if __name__ == "__main__":
    main()