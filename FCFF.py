import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import webbrowser

##################################################################################
###################################### Excel-Daten laden (Fundamental + Technical)

class FCFFModel:
    def __init__(self, stock="MSFT"):
        self.stock = stock
        self.all_stocks_data = self.load_stocks_data()
        self.fund_df = self.all_stocks_data[self.stock]["fundamental"]
        fy_dict = {
                        "MSFT": "FY 2024",
                        "NVDA": "FY 2025",
                    }
        self.chosen_fy = fy_dict[self.stock]
        self.bloomberg_data = {
            "base_year_revenue": self.get_value("Revenue", self.chosen_fy),
            "Cash": self.get_value("Cash, Cash Equivalents & STI", self.chosen_fy),
            "Total debt": self.get_value("Short Term Debt", self.chosen_fy) + self.get_value("Long Term Debt", self.chosen_fy),
            "Shares outstanding": self.get_value("Diluted Weighted Average Shares", self.chosen_fy),
            "base_date": self.get_reporting_date_for_fy(self.chosen_fy),
            "reporting_date": self.get_value_as_date("Latest Announcement Date", self.chosen_fy),   
            "ebit_balance": self.get_value("EBIT", self.chosen_fy),
            "invested_capital": self.get_value("Total Invested Capital",self.chosen_fy),
            "base_year_tax": self.get_value("Effective Tax Rate",self.chosen_fy)/100
                        }
        self.bloomberg_data["stock_price"] = float(self.all_stocks_data[self.stock]["technical"].loc[
            self.all_stocks_data[self.stock]["technical"].iloc[:, 0] == self.bloomberg_data["reporting_date"]
        ].iloc[0, 1])

        self.dates = self.generate_dates(self.bloomberg_data["base_date"].year)
        

    
    def load_stocks_data(self,folder="Stocks"):
        all_stocks_data = {}
        for f in Path(folder).glob("*.xlsx"):
            # Fundamental einlesen & transformieren
            df = pd.read_excel(f, sheet_name="Fundamental", header=[0, 1], skiprows=3)
            new_cols = [df.columns[0], df.columns[1]] + [f"{a}|{b}" for a, b in df.columns[2:]]
            df.columns = new_cols
            long_df = pd.melt(
                df,
                id_vars=[new_cols[0], new_cols[1]],
                value_vars=new_cols[2:],
                var_name="Period",
                value_name="Value"
            )
            long_df[['FY', 'Reporting Date']] = long_df['Period'].str.split('|', expand=True)
            long_df = long_df.rename(columns={new_cols[0]: "Field"})
            long_df = long_df.drop(columns=new_cols[1])
            long_df = long_df[["Field", "FY", "Reporting Date", "Value"]]

            # Technical einlesen & transformieren
            tdf = pd.read_excel(f, sheet_name="Price", header=None)
            technical_df = tdf.iloc[7:].copy()
            technical_df.columns = tdf.iloc[6]
            technical_df.iloc[:, 0] = pd.to_datetime(technical_df.iloc[:, 0], errors='coerce')

            # Speichern
            all_stocks_data[f.stem] = {
                "fundamental": long_df,
                "technical": technical_df
            }
        return all_stocks_data
    


    def get_value(self,field, fy):
        vals = self.fund_df.loc[(self.fund_df["Field"] == field) & (self.fund_df["FY"] == fy), "Value"].values
        return float(vals[0]) if len(vals) > 0 else None

    def get_reporting_date_for_fy(self,fy):
        vals = self.fund_df.loc[self.fund_df["FY"] == fy, "Reporting Date"].values
        return pd.to_datetime(vals[0], errors="coerce") if len(vals) > 0 else None

    def get_value_as_date(self,field, fy):
        vals = self.fund_df.loc[(self.fund_df["Field"] == field) & (self.fund_df["FY"] == fy), "Value"].values
        return pd.to_datetime(vals[0], errors="coerce") if len(vals) > 0 else None



    def generate_dates(self,base_year):
        dates = [datetime(base_year + i, self.bloomberg_data["base_date"].month, self.bloomberg_data["base_date"].day) for i in range(11)] #Richtiges Datum beibehlten
        dates.append("Terminal Value")
        return dates

    def build_forecast_df(self, user_inputs):
        revenues = [self.bloomberg_data["base_year_revenue"]]
        for growth in user_inputs["revenue_growth"]:
            revenues.append(revenues[-1] * (1 + growth))
        df = pd.DataFrame(index= self.dates)
        df["Revenue Growth"] = [np.nan] + user_inputs["revenue_growth"]
        df["Revenue"] = revenues
        df["Operating Margin"] = [self.bloomberg_data["ebit_balance"] / self.bloomberg_data["base_year_revenue"]] + user_inputs["operating_margin"]
        df["EBIT"] = [self.bloomberg_data["ebit_balance"]] + (df["Revenue"][1:12] * user_inputs["operating_margin"]).tolist()
        
        df["Tax Rate"] =[self.bloomberg_data["base_year_tax"]] + user_inputs["tax_rate"]
        df["EBIT after Tax"] = [np.nan] + (df["EBIT"][1:].values * (1 - np.array(user_inputs["tax_rate"][0:]))).tolist()
        df["Reinvestment Rate"] = [np.nan] + user_inputs["reinvestment_rate"] + [np.nan]
        df["Reinvestment"] = (
            (df["Revenue"].shift(-1) - df["Revenue"]) / df["Reinvestment Rate"]
        ).tolist()[:-1] + [
            df.at["Terminal Value", "EBIT after Tax"] * df.at["Terminal Value", "Revenue Growth"] / user_inputs["roic_tv"]
        ]
        df["FCFF"] = (df["EBIT after Tax"] - df["Reinvestment"]).iloc[:-1].tolist() + [np.nan]
        df["WACC"] = [np.nan] + user_inputs["wacc"]
        df["Discount Factor"] = ((1 + df["WACC"]).cumprod() ** -1).iloc[:-1].tolist() + [np.nan]
        df["Discounted FCFF"] = (df["FCFF"] * df["Discount Factor"]).iloc[:-1].tolist() + [np.nan]
        return df, df.T

    def calculate_valuation(self,user_inputs):
        x,y = self.build_forecast_df(user_inputs= user_inputs)
        results = {
            "Sum of Discounted FCFFs": np.nansum(x["Discounted FCFF"]),
            "TV - FCFF": x.at["Terminal Value", "EBIT after Tax"] - x.at["Terminal Value", "Reinvestment"],
            "Terminal Value": (
                x.at["Terminal Value", "EBIT after Tax"] - x.at["Terminal Value", "Reinvestment"]
            ) / (user_inputs["wacc"][-1] - user_inputs["revenue_growth"][-1]),
        }
        results["Discounted Terminal Value"] = (
            results["Terminal Value"] * x.at[x.index[-2], "Discount Factor"]
        )
        results["Total Firm Value"] = results["Sum of Discounted FCFFs"] + results["Discounted Terminal Value"]
        results["Equity Value"] = results["Total Firm Value"] - self.bloomberg_data["Total debt"] + self.bloomberg_data["Cash"]
        results["Shares Outstanding"] = self.bloomberg_data["Shares outstanding"]
        results["Fair Value per Share"] = results["Equity Value"] / self.bloomberg_data["Shares outstanding"]
        results[f"Market Price ({self.bloomberg_data['reporting_date']:%d.%m.%Y})"] = self.bloomberg_data["stock_price"]
        return pd.DataFrame.from_dict(results, orient="index", columns=["Value"])


    def build_roic_df(self, user_inputs):
        x,y = self.build_forecast_df(user_inputs= user_inputs)
        roic_df = pd.DataFrame(index=self.dates)
        invested = [self.bloomberg_data["invested_capital"]] + (
            x["Reinvestment"].iloc[1:-1].cumsum() + self.bloomberg_data["invested_capital"]
        ).tolist() + [np.nan]
        roic_df["Invested Capital"] = invested
        avg = [np.nan] + ((pd.Series(invested).shift(1) + pd.Series(invested)) / 2).iloc[1:-1].tolist() + ["After 10 Years:"]
        roic_df["Avg Invested Capital"] = avg
        roic_df["Return on Invested Capital"] = [np.nan] + (
            x["EBIT after Tax"][1:-1] / roic_df["Avg Invested Capital"][1:-1]
        ).tolist() + [user_inputs["roic_tv"]]
        return roic_df, roic_df.T



    def plot_stock_price(self):
        df = self.all_stocks_data[self.stock]["technical"].copy()
        # Wir nehmen die Spaltennamen dynamisch
        date_col = df.columns[0]  # z. B. 'Date'
        price_col = [c for c in df.columns if "px_last" in str(c).lower()][0]  # z. B. 'PX_LAST'

        df = df[pd.to_datetime(df[date_col]).dt.year >= 2012]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(df[date_col], df[price_col], linewidth=2.5)
        
        # Transparenz & Design
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_title(f"{self.stock}: Price History", fontsize=20, fontweight="bold", loc='left', pad=20)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Price", fontsize=14)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # X-Achse: nur Jahreszahlen als Ticks
        years = pd.DatetimeIndex(df[date_col]).year.unique()
        ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in years])
        ax.set_xticklabels([str(y) for y in years], fontsize=12)

        # Dynamisches Y-Limit mit etwas Rand
        y = df[price_col]
        ax.set_ylim(y.min() - (y.max()-y.min())*0.05, y.max() + (y.max()-y.min())*0.10)

        plt.tight_layout()
        ax.grid(False)
        return fig, ax
    
    def plot_revenue_and_growth(self):
        df = self.all_stocks_data[self.stock]["fundamental"].copy()
        # Filter: nur Revenue und Revenue Growth, und nur bis inkl. reporting_date
        df = df[df["Field"].isin(["Revenue", "Revenue Growth (%)"])]
        df["Reporting Date"] = pd.to_datetime(df["Reporting Date"], errors="coerce")
        df = df[pd.to_datetime(df["Reporting Date"]).dt.year >= 2012]
        df = df[df["Reporting Date"] <= self.bloomberg_data["reporting_date"]]

        # Pivot: pro Reporting Date eine Zeile, Spalten sind Field (Revenue und Revenue Growth)
        df_pivot = df.pivot(index="Reporting Date", columns="Field", values="Value").sort_index()

        # Hier: Werte in Milliarden umrechnen
        df_pivot["Revenue"] = df_pivot["Revenue"] / 1000

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        bars = ax.bar(df_pivot.index, df_pivot["Revenue"], width=60, color="#2077b4", alpha=0.9)

        # Transparenz & Design wie im ersten Plot
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_title(f"{self.stock}: Revenue & YoY Growth", fontsize=20, fontweight="bold", loc='left', pad=20)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Revenue (in Billions USD)", fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # X-Achse: nur Jahr
        years = df_pivot.index.year
        ax.set_xticks(df_pivot.index)
        ax.set_xticklabels([str(y) for y in years], fontsize=12)

        # Y-Limits
        y = df_pivot["Revenue"]
        ax.set_ylim(y.min() - (y.max()-y.min())*0.05, y.max() + (y.max()-y.min())*0.12)

        # Revenue Growth (%) über jedem Balken, mit Zeilenumbruch vor YoY
        growth = df_pivot["Revenue Growth (%)"]
        for rect, val in zip(bars, growth):
            if pd.notnull(val):
                ax.text(
                    rect.get_x() + rect.get_width()/2, rect.get_height() + y.max()*0.01,
                    f"{val:.1f}%\nYoY", ha='center', va='bottom',
                    fontsize=11, fontweight="bold", color="#2077b4"
                )

        plt.tight_layout()
        ax.grid(False)
        return fig, ax
    
    def plot_ebit(self):
        df = self.all_stocks_data[self.stock]["fundamental"].copy()
        df_a = df[df["Field"] == "EBIT - 1 Yr Growth"]
        # Nur EBIT und nur bis inkl. reporting_date
        df = df[df["Field"] == "EBIT"]
        df["Reporting Date"] = pd.to_datetime(df["Reporting Date"], errors="coerce")
        df = df[pd.to_datetime(df["Reporting Date"]).dt.year >= 2012]
        df_a = df_a[pd.to_datetime(df_a["Reporting Date"]).dt.year >= 2012]
        df = df[df["Reporting Date"] <= self.bloomberg_data["reporting_date"]]
        df = df.sort_values("Reporting Date")

        # Werte in Milliarden
        y = df["Value"].astype(float) / 1000  

        # Wachstum berechnen (in Prozent)
        df_a["Value"] = pd.to_numeric(df_a["Value"], errors="coerce")
        growth = df_a["Value"]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        bars = ax.bar(df["Reporting Date"], y, width=60, color="#2077b4", alpha=0.9)

        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_title(f"{self.stock}: EBIT History", fontsize=20, fontweight="bold", loc='left', pad=20)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("EBIT (in Billions USD)", fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # X-Achse: nur Jahr
        years = df["Reporting Date"].dt.year
        ax.set_xticks(df["Reporting Date"])
        ax.set_xticklabels([str(y) for y in years], fontsize=12)

        # Y-Limits
        ax.set_ylim(y.min() - (y.max()-y.min())*0.05, y.max() + (y.max()-y.min())*0.12)

        # Growth-Labels über jedem Balken (außer erster, weil dort NaN)
        for rect, g in zip(bars, growth):
            if pd.notnull(g):
                ax.text(
                    rect.get_x() + rect.get_width()/2,
                    rect.get_height() + y.max()*0.01,
                    f"{g:.1f}%\nYoY",
                    ha='center', va='bottom',
                    fontsize=11, fontweight="bold", color="#2077b4"
                )

        plt.tight_layout()
        ax.grid(False)
        return fig, ax
    
    def plot_operating_margin(self):
        df = self.all_stocks_data[self.stock]["fundamental"].copy()
        # Nur Operating Margin und nur bis inkl. reporting_date
        df = df[df["Field"] == "Operating Margin"]
        df["Reporting Date"] = pd.to_datetime(df["Reporting Date"], errors="coerce")
        df = df[pd.to_datetime(df["Reporting Date"]).dt.year >= 2012]
        df = df[df["Reporting Date"] <= self.bloomberg_data["reporting_date"]]
        df = df.sort_values("Reporting Date")

        # Wert als float (z. B. 0.417)
        y = df["Value"].astype(float)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        bars = ax.bar(df["Reporting Date"], y, width=60, color="#2077b4", alpha=0.9)

        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_title(f"{self.stock}: Operating Margin", fontsize=20, fontweight="bold", loc='left', pad=20)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Operating Margin (%)", fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # X-Achse: nur Jahr
        years = df["Reporting Date"].dt.year
        ax.set_xticks(df["Reporting Date"])
        ax.set_xticklabels([str(y) for y in years], fontsize=12)

        # Y-Limits
        ax.set_ylim(y.min() - (y.max()-y.min())*0.05, y.max() + (y.max()-y.min())*0.12)

        # Prozent-Label über jedem Balken
        for rect, val in zip(bars, y):
            if pd.notnull(val):
                ax.text(
                    rect.get_x() + rect.get_width()/2,
                    rect.get_height() + y.max()*0.01,
                    f"{val:.1f}%",
                    ha='center', va='bottom',
                    fontsize=11, fontweight="bold", color="#2077b4"
                )

        plt.tight_layout()
        ax.grid(False)
        return fig, ax
    

    def plot_invested_capital_and_roic(self):
        df = self.all_stocks_data[self.stock]["fundamental"].copy()
        df = df[df["Field"].isin(["Total Invested Capital", "Return on Invested Capital"])]
        df["Reporting Date"] = pd.to_datetime(df["Reporting Date"], errors="coerce")
        df = df[pd.to_datetime(df["Reporting Date"]).dt.year >= 2012]
        df = df[df["Reporting Date"] <= self.bloomberg_data["reporting_date"]]
        df = df.sort_values("Reporting Date")

        # Pivot, damit beide Werte pro Jahr nebeneinander stehen
        df_pivot = df.pivot(index="Reporting Date", columns="Field", values="Value").sort_index()
        y_bars = df_pivot["Total Invested Capital"].astype(float) / 1000  # in Milliarden
        y_line = df_pivot["Return on Invested Capital"].astype(float).round(0)  # in %, gerundet

        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=100)
        bars = ax1.bar(df_pivot.index, y_bars, width=60, color="#2077b4", alpha=0.9, label="Total Invested Capital")

        fig.patch.set_alpha(0)
        ax1.patch.set_alpha(0)
        ax1.set_title(f"{self.stock}: Invested Capital & ROIC", fontsize=20, fontweight="bold", loc='left', pad=20)
        ax1.set_xlabel("Year", fontsize=14)
        ax1.set_ylabel("Total Invested Capital (in Billions USD)", fontsize=14, color="#2077b4")
        ax1.tick_params(axis='y', labelcolor="#2077b4")
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        years = df_pivot.index.year
        ax1.set_xticks(df_pivot.index)
        ax1.set_xticklabels([str(y) for y in years], fontsize=12)
        ax1.set_ylim(y_bars.min() - (y_bars.max()-y_bars.min())*0.05, y_bars.max() + (y_bars.max()-y_bars.min())*0.15)
        ax1.grid(False)

        # Sekundärachse für ROIC, modernes Türkis
        accent_color = "#2bdab3"  # Komplementär zu Blau
        ax2 = ax1.twinx()
        ax2.plot(
            df_pivot.index, y_line, color=accent_color, linewidth=3.5, marker='o',
            markersize=9, label="ROIC"
        )
        ax2.set_ylabel("ROIC (%)", fontsize=14, color=accent_color)
        ax2.tick_params(axis='y', labelcolor=accent_color)
        ax2.set_ylim(
            y_line.min() - (y_line.max()-y_line.min())*0.05,
            y_line.max() + (y_line.max()-y_line.min())*0.15
        )
        ax2.spines['top'].set_visible(False)

        # Werte unter jedem ROIC-Punkt, klein und subtil
        for x, val in zip(df_pivot.index, y_line):
            if pd.notnull(val):
                ax2.text(
                    x, val + (y_line.max()*0.05),
                    f"{val:.0f}%",
                    ha='center', va='top',
                    fontsize=9, fontweight="normal", color="#807c7c"
                )

        plt.tight_layout()
        return fig, ax1, ax2
    

    def plot_reinvestment_only(self):
        # Daten holen und sortieren
        fundamental_data = self.all_stocks_data[self.stock]["fundamental"].copy()
        invested_capital_data = fundamental_data[fundamental_data["Field"] == "Total Invested Capital"].copy()
        invested_capital_data["Reporting Date"] = pd.to_datetime(invested_capital_data["Reporting Date"], errors="coerce")
        
        invested_capital_data = invested_capital_data[invested_capital_data["Reporting Date"] <= self.bloomberg_data["reporting_date"]]
        invested_capital_data = invested_capital_data.sort_values("Reporting Date")
        reporting_dates = invested_capital_data["Reporting Date"]
        reporting_dates =  reporting_dates[2:]
        total_invested_capital = invested_capital_data["Value"].astype(float) / 1000  # in Milliarden

        # Reinvestment (Jahresdifferenz)
        reinvestment = total_invested_capital.diff().fillna(0)
        reinvestment = reinvestment[2:]
        #reporting_dates = reporting_dates[pd.to_datetime(reporting_dates).dt.year >= 2012]
        # Dezente Farben
        green = "#2bdab3"
        red = "#e27373"

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

        # Positive Werte (Grün)
        ax.bar(
            reporting_dates[reinvestment >= 0],
            reinvestment[reinvestment >= 0],
            width=60, color=green
        )
        # Negative Werte (Rot)
        ax.bar(
            reporting_dates[reinvestment < 0],
            reinvestment[reinvestment < 0],
            width=60, color=red
        )

        # Cleanes Design wie in deinen anderen Plots
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        ax.set_title(f"{self.stock}: Annual Reinvestment", fontsize=20, fontweight="bold", loc='left', pad=20)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Reinvestment (in Billions USD)", fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        years = reporting_dates.dt.year
        ax.set_xticks(reporting_dates)
        ax.set_xticklabels([str(y) for y in years], fontsize=12)
        ax.grid(False)

        plt.tight_layout()
        return fig, ax
    
    def sankey_microsoft(self, segment_file="Segment/MSFT.xlsx"):
        def clean_field_name(x):
            return x.lstrip(" +-")
        
        def get_value(df, field):
            return df.loc[df["Field"] == field, "Value"].values[0]
        
        def get_color(row):
            from_, to_ = row['from'], row['to']

            # Grau für Segment-Zusammenfassungen und Revenue-Verbindungen
            if to_ in ["Revenue"]:
                return "rgba(110,110,110,0.5)"
            
            if to_ in ["Productivity & Business Processes", "Intelligent Cloud", "More Personal Computing"]:
                return "rgba(110,110,110,0.25)"

            # Grün für alle Gewinn-Flows
            if to_ in ["Gross Profit", "Operating Income", "Net Income"]:
                return "rgba(40,200,120,0.55)"

            # Pink für alles andere (Kosten, Verluste, Steuern, Expenses etc.)
            return "rgba(234,49,97,0.45)"
        
        def b_fmt(val):
            return f"${float(val)/1000:.1f}B"

        def f_val_yoy(field):
            row = fundamental_df.loc[fundamental_df["Field"] == field].iloc[0]
            val_fmt = b_fmt(row["Value"])
            yoy_fmt = f"({row['YoY']:+.1%})" if pd.notnull(row["YoY"]) else ""
            return f"{val_fmt}<br><span style='color:#888'>{yoy_fmt}</span>"

        def seg_val_yoy(segment):
            row = segment_df[segment_df.iloc[:, 0] == segment].iloc[0]
            val_fmt = b_fmt(row[2])
            yoy_fmt = f"({row['YoY']:+.1%})" if pd.notnull(row["YoY"]) else ""
            return f"{val_fmt}<br><span style='color:#888'>{yoy_fmt}</span>"


        def get_node_color(label):
            # HTML-Name extrahieren
            name = label.split("<")[1].split(">")[1] if "<b>" in label else label.split("<br>")[0]
            name = name.strip()
            # Segment-Farben (links)
            if name in segment_names or name in category_names:
                return "rgba(110,110,110,0.85)"  # dunkelgrau Segment
            # Profit/Income
            if "Profit" in name or "Income" in name or "Net" in name:
                return "rgba(40,200,120,0.85)"    # grün
            # Kosten oder Expense/Tax etc.
            if "Cost" in name or "Loss" in name or "Expense" in name or "Tax" in name or "R&D" in name or "S&M" in name or "cost" in name or "G&A" in name:
                return "rgba(234,49,97,0.85)"     # pink
            # Revenue
            if "Revenue" in name:
                return "rgba(110,110,110,0.85)"   # grau
            return "rgba(110,110,110,0.85)"        # blasses grau
        
        def fmt_pct(x):
            return f"{x:.1%}"
            

        relevant_fields = [
            "Revenue",                             # ok
            "Gross Profit",                        # ok
            "Operating Income",                    # statt "Operating Income (Loss)"
            "Net Income",                          # statt "Net Income, GAAP"
            "Income Tax Expense (Benefit)",        # ok
            "Non-Operating (Income) Loss",         # ok
            "Abnormal Losses (Gains)",             # ok
            "Cost of Revenue",                     # ok
            "Operating Expenses",                  # ok
            "Selling & Marketing Expense",         # statt "Selling & Marketing"
            "General and Administrative",          # statt "General & Administrative"
            "R&D Expense",                         # statt "Research & Development"
            "Gross Margin",                        # ok
            "Operating Margin",                    # ok
            "Profit Margin",                        # ok
            "Cost of Products Sold",
            "Cost of Services"
            ]

        fundamental_df = (
            self.all_stocks_data["MSFT"]["fundamental"]
            .assign(Field=lambda df: df["Field"].apply(clean_field_name))
        )
        fundamental_df = fundamental_df[fundamental_df["Field"].isin(relevant_fields)]

        pivot = (
            fundamental_df
            .pivot(index="Field", columns="FY", values="Value")
            .apply(pd.to_numeric, errors="coerce")
        )



        # Margins berechnen (angepasste Spaltennamen!)
        rev = float(pivot.loc["Revenue", "FY 2024"])
        pivot.loc["Gross Margin", "FY 2024"] = float(pivot.loc["Gross Profit", "FY 2024"]) / rev
        pivot.loc["Operating Margin", "FY 2024"] = float(pivot.loc["Operating Income", "FY 2024"]) / rev  # angepasst!
        pivot.loc["Profit Margin", "FY 2024"] = float(pivot.loc["Net Income", "FY 2024"]) / rev           # angepasst!

        fundamental_df = (
            pivot
            .reset_index()
            .melt(id_vars="Field", var_name="FY", value_name="Value")
        )

        fundamental_df["YoY"] = (
            fundamental_df.loc[fundamental_df["FY"] == "FY 2024"]
                .set_index("Field")["Value"]
                .div(
                    fundamental_df.loc[fundamental_df["FY"] == "FY 2023"]
                    .set_index("Field")["Value"]
                ) - 1
        ).reindex(fundamental_df["Field"]).values

        fundamental_df = fundamental_df[fundamental_df["FY"] == "FY 2024"].reset_index(drop=True)


        sankey_flows = [
            # Revenue-Split
            {"from": "Revenue", "to": "Gross Profit", "value": get_value(fundamental_df, "Gross Profit")},
            {"from": "Revenue", "to": "Cost of Revenue", "value": get_value(fundamental_df, "Cost of Revenue")},

            # Gross Profit zu Operating Income und Operating Expenses
            {"from": "Gross Profit", "to": "Operating Income", "value": get_value(fundamental_df, "Operating Income")},
            {"from": "Gross Profit", "to": "Operating Expenses", "value": get_value(fundamental_df, "Operating Expenses")},

            # Operating Expenses splitten
            {"from": "Operating Expenses", "to": "R&D Expense", "value": get_value(fundamental_df, "R&D Expense")},
            {"from": "Operating Expenses", "to": "Selling & Marketing Expense", "value": get_value(fundamental_df, "Selling & Marketing Expense")},
            {"from": "Operating Expenses", "to": "General and Administrative", "value": get_value(fundamental_df, "General and Administrative")},

            # Operating Income splitten
            {"from": "Operating Income", "to": "Net Income", "value": get_value(fundamental_df, "Net Income")},
            {"from": "Operating Income", "to": "Income Tax Expense (Benefit)", "value": get_value(fundamental_df, "Income Tax Expense (Benefit)")},
            {"from": "Operating Income", "to": "Other Loss", "value": get_value(fundamental_df, "Non-Operating (Income) Loss")},
            
            # Cost of Revenue aufteilen
            {"from": "Cost of Revenue", "to": "Cost of Products Sold", "value": get_value(fundamental_df, "Cost of Products Sold")},
            {"from": "Cost of Revenue", "to": "Cost of Services", "value": get_value(fundamental_df, "Cost of Services")}
        ]


        segment_df = (
            pd.read_excel(segment_file, header=2)
            .iloc[:126]
        )
        segment_df.iloc[:, 0] = segment_df.iloc[:, 0].str.lstrip()

        main_segments = [
            "Microsoft 365 Commercial Products & Cloud Services",
            "Linkedin",
            "Dynamics Products & Cloud Services",
            "Microsoft 365 Consumer Products & Cloud Services",
            "Server Products & Cloud Services",
            "Enterprise Services",
            "Other",
            "Gaming",
            "Windows & Devices",
            "Search Advertising",
            "More Personal Computing",
            "Intelligent Cloud",
            "Productivity & Business Processes"
        ]

        
            
        keep_cols = [segment_df.columns[0]] + [col for col in segment_df.columns if "2024" in str(col) or "2023" in str(col)]
        segment_df = segment_df.loc[segment_df.iloc[:, 0].isin(main_segments), keep_cols].reset_index(drop=True)
        segment_df["YoY"] = segment_df.iloc[:, 2] / segment_df.iloc[:, 1] - 1

        segment_flow_tuples = [
            # Productivity and Business Processes
            ("Microsoft 365 Commercial Products & Cloud Services", "Productivity & Business Processes"),
            ("Linkedin", "Productivity & Business Processes"),
            ("Dynamics Products & Cloud Services", "Productivity & Business Processes"),
            ("Microsoft 365 Consumer Products & Cloud Services", "Productivity & Business Processes"),

            # Intelligent Cloud
            ("Server Products & Cloud Services", "Intelligent Cloud"),
            ("Enterprise Services", "Intelligent Cloud"),
            ("Other", "Intelligent Cloud"),

            # More Personal Computing
            ("Gaming", "More Personal Computing"),
            ("Windows & Devices", "More Personal Computing"),
            ("Search Advertising", "More Personal Computing"),
            
            # Revenue
            ("Productivity & Business Processes", "Revenue"),
            ("Intelligent Cloud", "Revenue"),
            ("More Personal Computing", "Revenue"),
        ]

        segment_flows = [
            {"from": f, "to": t, "value": float(segment_df.loc[segment_df.iloc[:, 0] == f].iloc[0, 2])}
            for f, t in segment_flow_tuples
        ]



        sankey_df = pd.DataFrame(segment_flows + sankey_flows)

        labels = list(pd.unique(sankey_df[["from", "to"]].values.ravel("K")))
        label_index = {label: i for i, label in enumerate(labels)}
        sources = sankey_df["from"].map(label_index)
        targets = sankey_df["to"].map(label_index)
        values  = sankey_df["value"].astype(float)




        link_colors = sankey_df.apply(get_color, axis=1).tolist()
        x_pos = [
            0.12,   # Microsoft 365 Commercial Products & Cloud Services
            0.04,   # Linkedin
            0.02,   # Dynamics Products & Cloud Services
            0.01,   # Microsoft 365 Consumer Products & Cloud Services
            0.005,   # Server Products & Cloud Services
            0.05,   # Enterprise Services
            0.005,   # Other
            0.04,   # Gaming
            0.08,   # Windows & Devices
            0.12,   # Search Advertising
            0.25,   # Productivity & Business Processes
            0.25,   # Intelligent Cloud
            0.25,   # More Personal Computing
            0.45,   # Revenue
            0.6,  # Gross Profit
            0.68,  # Operating Expenses
            0.68,  # Operating Income
            0.6,  # Cost of Revenue
            0.85,  # R&D Expense
            0.85,  # Selling & Marketing Expense
            0.85,  # General and Administrative
            0.85,  # Net Income
            0.85,  # Income Tax Expense (Benefit)
            0.85,  # Other Loss
            0.75,  # Cost of Products Sold
            0.75,  # Cost of Services
        ]

        y_pos = [
            0.02,  # Microsoft 365 Commercial Products & Cloud Services
            0.11,  # Linkedin
            0.22,  # Dynamics Products & Cloud Services
            0.34,  # Microsoft 365 Consumer Products & Cloud Services
            0.49,  # Server Products & Cloud Services
            0.59,  # Enterprise Services
            0.62,  # Other
            0.73,  # Gaming
            0.79,  # Windows & Devices
            0.85,  # Search Advertising
            0.3,  # Productivity & Business Processes
            0.50,  # Intelligent Cloud
            0.70,  # More Personal Computing
            0.5,   # Revenue
            0.25,  # Gross Profit
            0.4,  # Operating Expenses
            0.15,  # Operating Income
            0.7,  # Cost of Revenue
            0.45,  # R&D Expense
            0.5,  # Selling & Marketing Expense
            0.55,  # General and Administrative
            0.02,  # Net Income
            0.06,  # Income Tax Expense (Benefit)
            0.09,  # Other Loss
            0.75,  # Cost of Products Sold
            0.79,  # Cost of Services
        ]

        labels = [
            "<b>Microsoft 365 Commercial<br>Products & Cloud Services</b><br>" + seg_val_yoy("Microsoft 365 Commercial Products & Cloud Services"),
            "<b>LinkedIn</b><br>" + seg_val_yoy("Linkedin"),
            "<b>Dynamics Products<br>& Cloud Services</b><br>" + seg_val_yoy("Dynamics Products & Cloud Services"),
            "<b>Microsoft 365 Consumer<br>Products & Cloud Services</b><br>" + seg_val_yoy("Microsoft 365 Consumer Products & Cloud Services"),
            "<b>Server Products<br>& Cloud Services</b><br>" + seg_val_yoy("Server Products & Cloud Services"),
            "<b>Enterprise Services</b><br>" + seg_val_yoy("Enterprise Services"),
            "<b>Other</b><br>" + seg_val_yoy("Other"),
            "<b>Gaming</b><br>" + seg_val_yoy("Gaming"),
            "<b>Windows & Devices</b><br>" + seg_val_yoy("Windows & Devices"),
            "<b>Search Advertising</b><br>" + seg_val_yoy("Search Advertising"),
            
            
            
            "<b>Productivity & Business Processes</b><br>" + seg_val_yoy("Productivity & Business Processes"),  
            "<b>Intelligent Cloud</b><br>" + seg_val_yoy("Intelligent Cloud"),
            "<b>More Personal Computing</b><br>" + seg_val_yoy("More Personal Computing"),


            "<b>Revenue</b><br>" + f_val_yoy("Revenue"),
            "<b>Gross Profit</b><br>" + f_val_yoy("Gross Profit"),
            
            "<b>Operating Expenses</b><br>" + f_val_yoy("Operating Expenses"),
        
            "<b>Operating Income</b><br>" + f_val_yoy("Operating Income"),
            
            "<b>Cost of Revenue</b><br>" + f_val_yoy("Cost of Revenue"),
            
        
            "<b>R&D Expense</b><br>" + f_val_yoy("R&D Expense"),
            "<b>S&M</b><br>" + f_val_yoy("Selling & Marketing Expense"),
            "<b>G&A</b><br>" + f_val_yoy("General and Administrative"),
            "<b>Net Income</b><br>" + f_val_yoy("Net Income"),
            "<b>Tax</b><br>" + f_val_yoy("Income Tax Expense (Benefit)"),
            "<b>Other Loss</b><br>" + f_val_yoy("Non-Operating (Income) Loss"),
            "<b>Product cost</b><br>" + f_val_yoy("Cost of Products Sold"),
            "<b>Service cost</b><br>" + f_val_yoy("Cost of Services"),
        ]


        segment_names = [
            "Microsoft 365 Commercial Products & Cloud Services",
            "Linkedin",
            "Dynamics Products & Cloud Services",
            "Microsoft 365 Consumer Products & Cloud Services",
            "Server Products & Cloud Services",
            "Enterprise Services",
            "Other",
            "Gaming",
            "Windows & Devices",
            "Search Advertising"
        ]

        category_names = [
            "Productivity & Business Processes",
            "Intelligent Cloud",
            "More Personal Computing"
        ]


        node_colors = [get_node_color(l) for l in labels]




        # Margins für Textbox
        gross_margin = float(pivot.loc["Gross Margin", "FY 2024"])
        operating_margin = float(pivot.loc["Operating Margin", "FY 2024"])
        profit_margin = float(pivot.loc["Profit Margin", "FY 2024"])

        textbox = (
            f"<b>FY 2025 Margins:</b><br>"
            f"Gross Margin: <b>{fmt_pct(gross_margin)}</b><br>"
            f"Operating Margin: <b>{fmt_pct(operating_margin)}</b><br>"
            f"Profit Margin: <b>{fmt_pct(profit_margin)}</b>"
        )

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=50,
                thickness=5,
                label=labels,
                color=node_colors,
                y=y_pos,
                x=x_pos,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])

        fig.add_annotation(
            x=0.45, y=0.05, xref="paper", yref="paper",
            text=textbox,
            showarrow=False,
            font=dict(size=20, color="black"),
            align="left",
            bgcolor="rgba(65, 105, 225, 0.12)",
            borderwidth=1,
            borderpad=8,
        )

        fig.update_layout(
            title_text="Microsoft FY 2024 Segment Breakdown",
            font=dict(size=16, family="Arial"),
            margin=dict(l=60, r=60, t=60, b=30),
            width=1600, height=900,
            paper_bgcolor="white"
        )

        fig.write_html("msftsankey.html")
        import webbrowser
        # webbrowser.open("msftsankey.html")

##################################################################################################################################
    
    def sankey_nvidia(self, segment_file="Segment/NVDA.xlsx"):
        def clean_field_name(x): return x.lstrip(" +-")

        def get_value(df, field): return df.loc[df["Field"] == field, "Value"].values[0]

        def get_color(row):
            from_, to_ = row['from'], row['to']
            if to_ == "Revenue":
                return "rgba(110,110,110,0.5)"
            elif (from_, to_) in [
                ("Revenue", "Gross Profit"),
                ("Gross Profit", "Operating Income"),
                ("Operating Income", "Net Income"),
                ("Other Income", "Net Income")
            ]:
                return "rgba(40,200,120,0.5)"
            elif to_ in ["Cost of Revenue", "Operating Expenses", "R&D", "Sales, G&A"]:
                return "rgba(234,49,97,0.45)"
            elif to_ in ["Income Tax Expense (Benefit)"]:
                return "rgba(234,49,97,0.45)"
            elif (from_, to_) in [
                ("Operating Expenses", "R&D Expense"),
                ("Operating Expenses", "Selling, General and Administrative Expense")
            ]:
                return "rgba(234,49,97,0.45)"
            else:
                return "rgba(180,180,180,0.25)"


        def b_fmt(val):
            return f"${float(val)/1000:.1f}B"

        def f_val(field):
            return b_fmt(fundamental_df.loc[fundamental_df["Field"] == field, "Value"].iloc[0])

        def f_val_yoy(field):
            row = fundamental_df.loc[fundamental_df["Field"] == field].iloc[0]
            val_fmt = b_fmt(row["Value"])
            yoy_fmt = f"({row['YoY']:+.1%})" if pd.notnull(row["YoY"]) else ""
            return f"{val_fmt}<br><span style='color:#888'>{yoy_fmt}</span>"

        def seg_val(name):
            return b_fmt(segment_df.loc[segment_df.iloc[:, 0] == name].iloc[0, 2])

        def seg_val_yoy(segment):
            row = segment_df[segment_df.iloc[:, 0] == segment].iloc[0]
            val_fmt = b_fmt(row[2])
            yoy_fmt = f"({row['YoY']:+.1%})" if pd.notnull(row["YoY"]) else ""
            return f"{val_fmt}<br><span style='color:#888'>{yoy_fmt}</span>"


        def get_node_color(label):
            # Extrahiere den Namen ohne HTML
            name = label.split("<")[1].split(">")[1] if "<b>" in label else label.split("<br>")[0]
            name = name.strip()
            if name in segment_names:
                return "rgba(110,110,110,0.85)"  # Grau für Segmente!
            if "Profit" in name or "Income" in name or "Net" in name:
                return "rgba(40,200,120,0.85)"    # Grün
            if "Cost" in name or "Expense" in name or "Tax" in name or "R&D" in name or "Selling" in name:
                return "rgba(234,49,97,0.85)"     # Pink
            if "Revenue" in name:
                return "rgba(110,110,110,0.85)"   # Grau
            return "rgba(180,180,180,0.7)"        # Blass grau


        def fmt_pct(x):
            return f"{x:.1%}"

        

        fundamental_df = self.all_stocks_data["NVDA"]["fundamental"].assign(Field=lambda df: df["Field"].apply(clean_field_name))
        relevant_fields = [
        "Revenue", "Gross Profit", "Operating Income", "Net Income", "Income Tax Expense (Benefit)",
        "Non-Operating (Income) Loss", "Abnormal Losses (Gains)", "Cost of Revenue", "Operating Expenses",
        "Selling, General and Administrative Expense", "R&D Expense", "Other Operating Expense",
        "Gross Margin", "Operating Margin", "Profit Margin", "Non-Operating (Income) Loss"
    ]
        

    
        fundamental_df = fundamental_df[fundamental_df["Field"].isin(relevant_fields)]

        pivot = (
            fundamental_df
            .pivot(index="Field", columns="FY", values="Value")
            .apply(pd.to_numeric, errors="coerce")
        )

        '''pivot.loc["Other Income"] = (
            np.where(pivot.loc["Abnormal Losses (Gains)"] < 0, np.abs(pivot.loc["Abnormal Losses (Gains)"]), 0)
            + np.where(pivot.loc["Non-Operating (Income) Loss"] < 0, np.abs(pivot.loc["Non-Operating (Income) Loss"]), 0)
        )'''

        # Margins berechnen und als neue Felder hinzufügen
        rev = float(pivot.loc["Revenue", "FY 2025"])
        pivot.loc["Gross Margin", "FY 2025"] = float(pivot.loc["Gross Profit", "FY 2025"]) / rev
        pivot.loc["Operating Margin", "FY 2025"] = float(pivot.loc["Operating Income", "FY 2025"]) / rev
        pivot.loc["Profit Margin", "FY 2025"] = float(pivot.loc["Net Income", "FY 2025"]) / rev

        fundamental_df = (
            pivot
            .reset_index()
            .melt(id_vars="Field", var_name="FY", value_name="Value")
        )

        fundamental_df["YoY"] = (
            fundamental_df.loc[fundamental_df["FY"] == "FY 2025"]
                .set_index("Field")["Value"]
                .div(
                    fundamental_df.loc[fundamental_df["FY"] == "FY 2024"]
                    .set_index("Field")["Value"]
                ) - 1
        ).reindex(fundamental_df["Field"]).values

        fundamental_df = fundamental_df[fundamental_df["FY"] == "FY 2025"].reset_index(drop=True)
        # Helper-Funktionen
        def get_value(df, field):
            return df.loc[df["Field"] == field, "Value"].values[0]

        sankey_flows = [
            {"from": "Revenue", "to": "Gross Profit", "value": get_value(fundamental_df, "Gross Profit")},
            {"from": "Revenue", "to": "Cost of Revenue", "value": get_value(fundamental_df, "Cost of Revenue")},
            {"from": "Gross Profit", "to": "Operating Income", "value": get_value(fundamental_df, "Operating Income")},
            {"from": "Gross Profit", "to": "Operating Expenses", "value": get_value(fundamental_df, "Operating Expenses")}, #+ get_value(fundamental_df, "Other Operating Expense")},
            {"from": "Operating Expenses", "to": "Selling, General and Administrative Expense", "value": get_value(fundamental_df, "Selling, General and Administrative Expense")},
            {"from": "Operating Expenses", "to": "R&D Expense", "value": get_value(fundamental_df, "R&D Expense")},
            #{"from": "Operating Expenses", "to": "Other Operating Expense", "value": get_value(fundamental_df, "Other Operating Expense")},
            {"from": "Operating Income", "to": "Net Income", "value": get_value(fundamental_df, "Net Income")},
            {"from": "Operating Income", "to": "Income Tax Expense (Benefit)", "value": get_value(fundamental_df, "Income Tax Expense (Benefit)")},
            {"from": "Other Income", "to": "Net Income", "value":(-1)*get_value(fundamental_df, "Non-Operating (Income) Loss")}, #Minus weil negativ Loss =Income
        ]

        segment_df = (
            pd.read_excel(segment_file, header=3)
            .iloc[:17]
        )
        segment_df.iloc[:, 0] = segment_df.iloc[:, 0].str.lstrip()
        main_segments = [
            "Gaming", "Professional Visualization", "Data Center",
            "Compute", "Networking", "Automotive", "OEM & Other"
        ]
        keep_cols = [segment_df.columns[0]] + [col for col in segment_df.columns if "2024" in str(col) or "2025" in str(col)]
        segment_df = segment_df.loc[segment_df.iloc[:, 0].isin(main_segments), keep_cols].reset_index(drop=True)
        segment_df["YoY"] = segment_df.iloc[:, 2] / segment_df.iloc[:, 1] - 1

        segment_flow_tuples = [
            ("Compute", "Data Center"), ("Networking", "Data Center"),
            ("Data Center", "Revenue"), ("Gaming", "Revenue"),
            ("Professional Visualization", "Revenue"), ("Automotive", "Revenue"),
            ("OEM & Other", "Revenue"),
        ]
        segment_flows = [
            {"from": f, "to": t, "value": float(segment_df.loc[segment_df.iloc[:, 0] == f].iloc[0, 2])}
            for f, t in segment_flow_tuples
        ]

        sankey_df = pd.DataFrame(segment_flows + sankey_flows)

        labels = [
            "Compute",
            "Networking",
            "Gaming",
            "Professional Visualization",
            "Automotive",
            "OEM & Other",
            "Data Center",
            "Revenue",
            "Gross Profit",
            "Cost of Revenue",
            "Operating Income",
            "Operating Expenses",
            "Net Income",
            "Income Tax Expense (Benefit)",
            "Other Income",
            "Selling, General and Administrative Expense",
            "R&D Expense",
            "Other Operating Expense"
        ]

        label_index = {l: i for i, l in enumerate(labels)}
        sources = sankey_df["from"].map(label_index)
        targets = sankey_df["to"].map(label_index)
        values = sankey_df["value"].astype(float)

        link_colors = sankey_df.apply(get_color, axis=1).tolist()

        y_pos = [
            0.05, 0.09, 0.26, 0.28, 0.30, 0.32, 0.07, 0.4, 0.25, 0.55,
            0.15, 0.26, 0.10, 0.13, 0.02, 0.35, 0.39, 0.55
        ]
        x_pos = [
            0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.16, 0.4, 0.57, 0.57,
            0.65, 0.65, 0.84, 0.84, 0.7, 0.84, 0.84, 0.84
        ]

        # Für die Labels (Segment- und GuV mit YoY etc.)
        def b_fmt(val):
            return f"${float(val)/1000:.1f}B"

        def f_val_yoy(field):
            row = fundamental_df.loc[fundamental_df["Field"] == field].iloc[0]
            val_fmt = b_fmt(row["Value"])
            yoy_fmt = f"({row['YoY']:+.1%})" if pd.notnull(row["YoY"]) else ""
            return f"{val_fmt}<br><span style='color:#888'>{yoy_fmt}</span>"

        def seg_val_yoy(segment):
            row = segment_df[segment_df.iloc[:, 0] == segment].iloc[0]
            val_fmt = b_fmt(row[2])
            yoy_fmt = f"({row['YoY']:+.1%})" if pd.notnull(row["YoY"]) else ""
            return f"{val_fmt}<br><span style='color:#888'>{yoy_fmt}</span>"

        labels = [
            "<b>Compute</b><br>" + seg_val_yoy("Compute"),
            "<b>Networking</b><br>" + seg_val_yoy("Networking"),
            "<b>Gaming</b><br>" + seg_val_yoy("Gaming"),
            "<b>Professional Visualization</b><br>" + seg_val_yoy("Professional Visualization"),
            "<b>Automotive</b><br>" + seg_val_yoy("Automotive"),
            "<b>OEM & Other</b><br>" + seg_val_yoy("OEM & Other"),
            "<b>Data Center</b><br>" + seg_val_yoy("Data Center"),
            "<b>Revenue</b><br>" + f_val_yoy("Revenue"),
            "<b>Gross Profit</b><br>" + f_val_yoy("Gross Profit"),
            "<b>Cost of Revenue</b><br>" + f_val_yoy("Cost of Revenue"),
            "<b>Operating Income</b><br>" + f_val_yoy("Operating Income"),
            "<b>Operating Expenses</b><br>" + f_val_yoy("Operating Expenses"),
            "<b>Net Income</b><br>" + f_val_yoy("Net Income"),
            "<b>Tax</b><br>" + f_val_yoy("Income Tax Expense (Benefit)"),
            "<b>Other Income</b><br>" + f_val_yoy("Non-Operating (Income) Loss"),
            "<b>Sales, G&A</b><br>" + f_val_yoy("Selling, General and Administrative Expense"),
            "<b>R&D</b><br>" + f_val_yoy("R&D Expense")
            #"<b>Other Operating Expense</b><br>" + f_val_yoy("Other Operating Expense")
        ]

        segment_names = [
            "Compute", "Networking", "Gaming", "Professional Visualization",
            "Automotive", "OEM & Other", "Data Center"
        ]

        def get_node_color(label):
            name = label.split("<")[1].split(">")[1] if "<b>" in label else label.split("<br>")[0]
            name = name.strip()
            if name in segment_names:
                return "rgba(110,110,110,0.85)"  # Grau für Segmente!
            if "Profit" in name or "Income" in name or "Net" in name:
                return "rgba(40,200,120,0.85)"    # Grün
            if "Cost" in name or "Expense" in name or "Sales" in name or "Tax" in name or "R&D" in name or "Selling" in name:
                return "rgba(234,49,97,0.85)"     # Pink
            if "Revenue" in name:
                return "rgba(110,110,110,0.85)"   # Grau
            return "rgba(180,180,180,0.7)"        # Blass grau

        node_colors = [get_node_color(l) for l in labels]

        # Margins für Textbox
        gross_margin = float(pivot.loc["Gross Margin", "FY 2025"])
        operating_margin = float(pivot.loc["Operating Margin", "FY 2025"])
        profit_margin = float(pivot.loc["Profit Margin", "FY 2025"])

        def fmt_pct(x):
            return f"{x:.1%}"

        textbox = (
            f"<b>FY 2025 Margins:</b><br>"
            f"Gross Margin: <b>{fmt_pct(gross_margin)}</b><br>"
            f"Operating Margin: <b>{fmt_pct(operating_margin)}</b><br>"
            f"Profit Margin: <b>{fmt_pct(profit_margin)}</b>"
        )

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=90,
                thickness=10,
                label=labels,
                color=node_colors,
                y=y_pos,
                x=x_pos
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])

        fig.add_annotation(
            x=0.84, y=0.10, xref="paper", yref="paper",
            text=textbox,
            showarrow=False,
            font=dict(size=20, color="black"),
            align="left",
            bgcolor="rgba(65, 105, 225, 0.12)",
            borderwidth=1,
            borderpad=8,
        )

        fig.update_layout(
            title_text="NVIDIA FY25 Segment Breakdown",
            font=dict(size=16, family="Arial"),
            margin=dict(l=60, r=60, t=60, b=30),
            width=1600, height=900,
            paper_bgcolor="white"
        )

        fig.write_html("nvda_sankey_final.html")
        import webbrowser
        # webbrowser.open("nvda_sankey_final.html")








    #################################################
    
