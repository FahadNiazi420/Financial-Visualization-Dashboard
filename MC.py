import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from FCFF import FCFFModel

class MonteCarloInputSimulator:
    def __init__(self, n_iter=1000, tax_rate=0.182):
        self.n_iter = n_iter
        self.tax_rate = tax_rate  # direkt gesetzt
        
        # Default-Parameter (alle steuerbar mit set_param)
        self.params = {
            "revenue_growth": {
                "mean": 0.30,
                "std": 0.02,
                "terminal_mean": 0.043,
                "terminal_std": 0.002,
                "structured": True,
                "clip_min": -1.0,
                "clip_max": 1
            },
            "operating_margin": {
                "mean": 0.40, "std": 0.03,
                "terminal_mean": 0.4, "terminal_std": 0.03,
                "clip_min": -1.0, "clip_max": 1
            },
            "reinvestment_rate": {
                "mean": 2.0, "std": 0.2, "size": 10,
                "clip_min": 0.0, "clip_max": 3.0
            },
            "wacc": {
                "mean": 0.09, "std": 0.005,
                "terminal_mean": 0.084, "terminal_std": 0.002,
                "clip_min": 0.01, "clip_max": 0.99
            },
            "roic_tv": {
                "mean": 0.2, "std": 0.02,
                "clip_min": 0.1, "clip_max": 0.3
            }
        }
        self.fcff = FCFFModel( stock = 'NVDA')
        self.sim_inputs = self.simulate_all()
        self.fair_values = self.calculate_mc()


#Chat GPT
    def set_param(self, key, **kwargs):
        if key not in self.params:
            self.params[key] = {}
        self.params[key].update(kwargs)


    def _simulate_structured_growth(self):
        cfg = self.params["revenue_growth"]
        
        g1 = np.random.normal(cfg["mean"], cfg["std"])
        g_terminal = np.random.normal(cfg["terminal_mean"], cfg["terminal_std"])

        # Clip beide Werte:
        g1 = np.clip(g1, cfg.get("clip_min", -np.inf), cfg.get("clip_max", np.inf))
        g_terminal = np.clip(g_terminal, cfg.get("clip_min", -np.inf), cfg.get("clip_max", np.inf))

        phase1 = [g1] * 5
        phase2 = np.linspace(g1, g_terminal, 6)[1:]  # Jahr 6–9
        return phase1 + phase2.tolist() + [g_terminal]


    def simulate_all(self):
        simulations = []
        for _ in range(self.n_iter):
            sim_input = {}

            for key, cfg in self.params.items():
                if key == "revenue_growth" and cfg.get("structured", False):
                    sim_input[key] = self._simulate_structured_growth()
                    continue

                # Sonderfall: wacc, operating_margin mit Terminalwert
                if key in ["wacc", "operating_margin"]:
                    main_vals = np.random.normal(cfg["mean"], cfg["std"], 10)
                    main_vals = np.clip(main_vals, cfg.get("clip_min", -np.inf), cfg.get("clip_max", np.inf))
                    terminal = np.random.normal(cfg["terminal_mean"], cfg["terminal_std"])
                    terminal = np.clip(terminal, cfg.get("clip_min", -np.inf), cfg.get("clip_max", np.inf))
                    sim_input[key] = main_vals.tolist() + [terminal]
                    continue

                # Sonderfall: reinvestment_rate → lognormal, kein Terminalwert
                if key == "reinvestment_rate":
                    val = np.random.lognormal(mean=np.log(cfg["mean"]), sigma=cfg["std"], size=cfg["size"])
                    val = np.clip(val, cfg.get("clip_min", -np.inf), cfg.get("clip_max", np.inf))
                    sim_input[key] = val.tolist()
                    continue

                # Standard: Einzelwert (z. B. roic_tv)
                mean = cfg["mean"]
                std = cfg["std"]
                val = np.random.normal(loc=mean, scale=std)
                if "clip_min" in cfg or "clip_max" in cfg:
                    val = np.clip(val, cfg.get("clip_min", -np.inf), cfg.get("clip_max", np.inf))
                sim_input[key] = float(val)

            # Steuerquote konstant aus Konstruktor
            sim_input["tax_rate"] = [self.tax_rate] * 11

            simulations.append(sim_input)

        return simulations


    def plot_fair_value_distribution(self, fair_values, stock_name="NVDA", stock_price=None):
    

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

        # Statistik
        fair_values_np = np.array(fair_values)
        mean_val = fair_values_np.mean()
        lower, upper = np.percentile(fair_values_np, [2.5, 97.5])

        shortfall = mean_excess_loss = None
        if stock_price is not None:
            below = fair_values_np[fair_values_np < stock_price]
            shortfall = np.mean(fair_values_np < stock_price) * 100
            mean_excess_loss = stock_price - below.mean() if len(below) > 0 else 0

        # Histogramm
        counts, bins, patches = ax.hist(
            fair_values,
            bins=50,
            color="#2077b4",
            alpha=0.85,
            edgecolor="white",
            density=True
        )

        # Achsen & Stil
        color = "#bdbdbd"
        ax.set_xlabel("Fair Value per Share", fontsize=14, color=color)
        ax.set_ylabel("Probability Density", fontsize=14, color=color)
        ax.tick_params(axis='x', labelcolor=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.spines['bottom'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # Titel
        ax.set_title(
            f"{stock_name}: Distribution of Fair Value per Share",
            fontsize=16, fontweight="bold", loc="left", pad=20
        )

        # Linien
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5,
                label=f"Ø Fair Value = {mean_val:.2f}")
        ax.axvspan(lower, upper, color="#2bdab3", alpha=0.2,
                label="95% Konfidenzintervall")
        if stock_price is not None:
            ax.axvline(stock_price, color="red", linestyle="--", linewidth=1.5,
                    label=f"Marktpreis = {stock_price:.2f}")

        # Simulation Count (einfach oben rechts, ohne Box)
        ax.text(
            0.95, 1.02,
            f"{self.n_iter} simulations",
            transform=ax.transAxes,
            fontsize=10,
            ha='right',
            va='top',
            color='gray'
        )

        # Info-Box rechts oben (Shortfall + MEL)
        info_lines = []
        if shortfall is not None:
            info_lines.append(f"Shortfall Risk     : {shortfall:>5.1f}%")
        if mean_excess_loss is not None:
            info_lines.append(f"Mean Excess Loss   : {mean_excess_loss:>5.2f}$")
        if info_lines:
            ax.text(
                0.95, 0.92,
                "\n".join(info_lines),
                transform=ax.transAxes,
                fontsize=10,
                ha='right',
                va='top',
                color='black',
                multialignment='left',
                bbox=dict(facecolor="#f2f2f2", edgecolor='none', alpha=1.0, boxstyle="round,pad=0.3")
            )

        # Legende links oben mit dezentem grauem Hintergrund
        legend = ax.legend(
            loc="upper left",
            fontsize=10,
            frameon=True
        )
        legend.get_frame().set_facecolor("#f2f2f2")
        legend.get_frame().set_edgecolor("none")

        plt.tight_layout()
        return fig, ax





    def plot_percentile_table(self, fair_values, steps=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            percentiles = np.percentile(fair_values, steps)
            data = [[f"{s}%", f"${v:.2f}"] for s, v in zip(steps, percentiles)]

            fig, ax = plt.subplots(figsize=(5, len(data) * 0.5), dpi=100)
            ax.axis("off")
            table = ax.table(
                cellText=data,
                colLabels=["Percentile", "Value Per Share"],
                cellLoc="center",
                loc="center"
            )

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)

            fig.suptitle("Percentile Table of Fair Value per Share", fontsize=14, fontweight="bold", y=0.95)
            plt.tight_layout()
            return fig, ax
    
    def calculate_mc(self):
        fair_values = []
        for user_input in self.sim_inputs:
            self.dates = self.fcff.generate_dates(self.fcff.bloomberg_data["base_date"].year)
            val_df = self.fcff.calculate_valuation(user_inputs=user_input)
            fair_values.append(val_df.loc["Fair Value per Share", "Value"])
        return fair_values