from experiments.toy_experiment import ToyConfig, run_toy


def main() -> None:
    cfg = ToyConfig(
        budgets=[64, 128, 256, 512, 1024],
        seeds=[0, 1, 2],
        interfaces=["argmax", "top2", "top3", "top5", "probs"],
    )
    df = run_toy(cfg)
    df.to_csv("results/toy_results_raw.csv", index=False)
    print(df.head())


if __name__ == "__main__":
    main()
