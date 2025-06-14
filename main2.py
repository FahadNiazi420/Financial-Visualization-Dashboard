import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtGui import QFont, QPixmap, QIntValidator
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QLineEdit, QTextEdit, QGroupBox, QScrollArea, QMessageBox, QSizePolicy, QGridLayout
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import your calculation modules
from FCFF import FCFFModel
from MC import MonteCarloInputSimulator

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots in PyQt5"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class ReverseFCFFTab(QWidget):
    """Tab 1: Reverse FCFF Tool with complete valuation table"""
    def __init__(self, parent=None):
        super(ReverseFCFFTab, self).__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout()
        self.setup_ui()
        
        # Initialize with default company
        self.update_company()
    
    def setup_ui(self):
        """Setup all UI components"""
        self.setup_header()
        self.setup_company_selection()
        self.setup_instruction_label()
        self.create_valuation_table()
        self.setup_calculate_button()
        self.setup_value_display()
        self.setLayout(self.layout)
    
    def setup_header(self):
        """Setup the header section"""
        header = QLabel("Reverse FCFF Tool")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(header)
    
    def setup_company_selection(self):
        """Setup company selection dropdown"""
        self.company_combo = QComboBox()
        self.company_combo.addItems(["Microsoft", "NVIDIA"])
        self.company_combo.currentTextChanged.connect(self.update_company)
        
        company_layout = QHBoxLayout()
        company_layout.addWidget(QLabel("Select Company:"))
        company_layout.addWidget(self.company_combo)
        company_layout.addStretch()
        self.layout.addLayout(company_layout)
    
    def setup_instruction_label(self):
        """Setup the instruction label"""
        instruction = QLabel("Please enter your valuation assumptions below and click the button to calculate the fair value.")
        instruction.setWordWrap(True)
        self.layout.addWidget(instruction)
    
    def create_valuation_table(self):
        """Create the comprehensive valuation table"""
        self.table = QTableWidget()
        self.table.setColumnCount(13)  # Base year + 10 years + terminal value
        self.table.setRowCount(23)     # All valuation metrics + ROIC after 10 years
        
        # Set headers
        headers = ["", "Baseyear : FY\n30.06.2024"] + \
                 [f"{i+1}. Year (30.06.{2025+i})" for i in range(10)] + \
                 ["Terminal Value"]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Set row labels
        row_labels = [
            "Revenue Growth",
            "Revenue",
            "Operating Margin",
            "EBIT",
            "Tax Rate",
            "EBT after Tax",
            "Reinvestment Rate",
            "Reinvestment",
            "FCFF",
            "WACC",
            "Discount Factor",
            "Discounted FCFF",
            "Sum of Discounted FCFF",
            "TV-FCFF",
            "Terminal Value = TV-FCFF/(k-g)",
            "Discounted Terminal Value",
            "Total Firm Value",
            "Total Debt",
            "plus Cash",
            "Equity Value",
            "Shares Outstanding",
            "Fair Value per Share",
            "Return on Invested Capital"
        ]
        self.table.setVerticalHeaderLabels(row_labels)
        
        # Configure editable cells
        editable_rows = {
            "Revenue Growth": range(1, 12),  # Years 1-10 + terminal
            "Operating Margin": range(1, 12),
            "Tax Rate": range(1, 12),
            "Reinvestment Rate": range(1, 11),  # Only years 1-10
            "WACC": range(1, 12)
        }
        
        for row in range(self.table.rowCount()):
            row_name = row_labels[row]
            for col in range(self.table.columnCount()):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                # Make cells editable based on row type
                if row_name in editable_rows and col in editable_rows[row_name]:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                
                self.table.setItem(row, col, item)
        
        self.table.resizeColumnsToContents()
        self.layout.addWidget(self.table)
    
    def setup_calculate_button(self):
        """Setup the calculate button"""
        self.calculate_btn = QPushButton("Calculate Fair Value")
        self.calculate_btn.clicked.connect(self.calculate_fair_value)
        self.layout.addWidget(self.calculate_btn)
    
    def setup_value_display(self):
        """Setup the fair value and market price display"""
        self.fair_value_label = QLabel("")
        self.fair_value_label.setStyleSheet("font-weight: bold; color: #2077b4")
        self.market_price_label = QLabel("")
        self.market_price_label.setStyleSheet("font-weight: bold; color: #2077b4")
        
        value_layout = QHBoxLayout()
        value_layout.addWidget(QLabel("Fair Value per Share:"))
        value_layout.addWidget(self.fair_value_label)
        value_layout.addWidget(QLabel("Market Price:"))
        value_layout.addWidget(self.market_price_label)
        value_layout.addStretch()
        self.layout.addLayout(value_layout)
    
    def update_company(self):
        """Update the table with default values for the selected company"""
        company = self.company_combo.currentText()
        stock = "MSFT" if company == "Microsoft" else "NVDA"
        
        # Update table header
        self.table.horizontalHeaderItem(0).setText(f"{company} ({stock} US)")
        
        # Initialize FCFF model to load data from files
        self.fcff = FCFFModel(stock=stock)
        
        # Set market price
        market_price = self.fcff.bloomberg_data["stock_price"]
        reporting_date = self.fcff.bloomberg_data["reporting_date"]
        self.market_price_label.setText(f"{market_price:.2f} ({reporting_date:%d.%m.%Y})")
        
        # Load default assumptions from the model
        self.load_default_assumptions(stock)
        
        # Set base year values from loaded data
        self.set_base_year_values()
        
        # Update parent's stock selection
        self.parent.update_stock(stock)
    
    def load_default_assumptions(self, stock):
        """Load default assumptions based on the selected stock"""
        if stock == "NVDA":
            defaults = {
                "revenue_growth": [0.10, 0.10, 0.10, 0.10, 0.10, 0.0986, 0.0972, 0.0858, 0.0644, 0.043],
                "operating_margin": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                "tax_rate": [0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142],
                "reinvestment_rate": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "wacc": [0.098, 0.098, 0.098, 0.098, 0.098, 0.0952, 0.0924, 0.0896, 0.0868, 0.0840],
                "roic_tv": 0.2
            }
        else:  # MSFT
            defaults = {
                "revenue_growth": [0.15, 0.15, 0.15, 0.15, 0.15, 0.1286, 0.1072, 0.0858, 0.0644, 0.043],
                "operating_margin": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                "tax_rate": [0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182],
                "reinvestment_rate": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "wacc": [0.098, 0.098, 0.098, 0.098, 0.098, 0.0952, 0.0924, 0.0896, 0.0868, 0.0840],
                "roic_tv": 0.2
            }
        
        # Map the defaults to our table rows
        row_mapping = {
            "revenue_growth": "Revenue Growth",
            "operating_margin": "Operating Margin",
            "tax_rate": "Tax Rate",
            "reinvestment_rate": "Reinvestment Rate",
            "wacc": "WACC"
        }
        
        # Find the row indices for each parameter
        row_indices = {}
        for row in range(self.table.rowCount()):
            row_name = self.table.verticalHeaderItem(row).text()
            for param, label in row_mapping.items():
                if row_name == label:
                    row_indices[param] = row
                    break
        
        # Populate the table with defaults
        for param, row in row_indices.items():
            values = defaults[param]
            for col in range(2, 12):  # Years 1-10 (columns 2-11)
                if param == "reinvestment_rate" and col == 11:
                    continue  # Skip terminal value for reinvestment rate
                
                value = values[col-2]  # Get the corresponding year's value
                if param in ["revenue_growth", "operating_margin", "tax_rate"]:
                    self.table.item(row, col).setText(f"{value:.2%}")
                elif param == "wacc":
                    self.table.item(row, col).setText(f"{value:.4f}")
                else:
                    self.table.item(row, col).setText(f"{value:.2f}")
        
        # Set terminal value WACC (same as year 10)
        if "wacc" in row_indices:
            terminal_wacc = defaults["wacc"][-1]
            self.table.item(row_indices["wacc"], 12).setText(f"{terminal_wacc:.4f}")
        
        # Set terminal ROIC (after 10 years)
        roic_row = self.find_row("Return on Invested Capital")
        if roic_row >= 0:
            self.table.item(roic_row, 12).setText(f"{defaults['roic_tv']:.2%}")
    
    def set_base_year_values(self):
        """Set the base year values from the loaded FCFF model data"""
        base_values = {
            "Revenue": self.fcff.bloomberg_data["base_year_revenue"] / 1000,  # Convert to billions
            "Operating Margin": self.fcff.bloomberg_data["ebit_balance"] / self.fcff.bloomberg_data["base_year_revenue"],
            "EBIT": self.fcff.bloomberg_data["ebit_balance"] / 1000,  # Convert to billions
            "Tax Rate": self.fcff.bloomberg_data["base_year_tax"],
            "Total Debt": self.fcff.bloomberg_data["Total debt"] / 1000,  # Convert to billions
            "plus Cash": self.fcff.bloomberg_data["Cash"] / 1000,  # Convert to billions
            "Shares Outstanding": self.fcff.bloomberg_data["Shares outstanding"] / 1000  # Convert to billions
        }
        
        # Set base year column (column 1)
        for row in range(self.table.rowCount()):
            row_name = self.table.verticalHeaderItem(row).text()
            if row_name == "Revenue":
                self.table.item(row, 1).setText(f"{base_values['Revenue']:,.2f}")
            elif row_name == "Operating Margin":
                self.table.item(row, 1).setText(f"{base_values['Operating Margin']:.2%}")
            elif row_name == "EBIT":
                self.table.item(row, 1).setText(f"{base_values['EBIT']:,.2f}")
            elif row_name == "Tax Rate":
                self.table.item(row, 1).setText(f"{base_values['Tax Rate']:.2%}")
            elif row_name == "Total Debt":
                self.table.item(row, 1).setText(f"{base_values['Total Debt']:,.2f}")
            elif row_name == "plus Cash":
                self.table.item(row, 1).setText(f"{base_values['plus Cash']:,.2f}")
            elif row_name == "Shares Outstanding":
                self.table.item(row, 1).setText(f"{base_values['Shares Outstanding']:,.2f}")
    
    def calculate_fair_value(self):
        """Calculate fair value based on user inputs and update the table"""
        try:
            # Get user inputs from table
            user_input = {
                "revenue_growth": self.parse_percentage_row("Revenue Growth"),
                "operating_margin": self.parse_percentage_row("Operating Margin"),
                "tax_rate": self.parse_percentage_row("Tax Rate"),
                "reinvestment_rate": self.parse_float_row("Reinvestment Rate"),
                "wacc": self.parse_float_row("WACC", include_terminal=True),
                "roic_tv": float(self.table.item(self.find_row("Return on Invested Capital"), 12).text().replace('%', '')) / 100
            }
            
            # 1. Build forecast dataframe
            forecast_df = self.fcff.build_forecast_df(user_inputs=user_input)[1]  # Get transposed version
            
            # 2. Calculate valuation
            valuation = self.fcff.calculate_valuation(user_inputs=user_input)
            
            # 3. Build ROIC dataframe
            roic_df = self.fcff.build_roic_df(user_inputs=user_input)[1]  # Get transposed version
            
            # Update the table with calculated values
            self.update_calculated_values(forecast_df, roic_df, valuation)
            
            # Display the fair value
            self.fair_value_label.setText(f"{valuation.loc['Fair Value per Share', 'Value']:,.2f}")
            
            # Update parent with new calculations
            self.parent.update_calculations(forecast_df, valuation, roic_df)
            
        except Exception as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input values: {str(e)}")
    
    def parse_percentage_row(self, row_name):
        """Parse a row of percentage values from the table"""
        row = self.find_row(row_name)
        values = []
        for col in range(2, 12):  # Years 1-10 + terminal
            text = self.table.item(row, col).text().replace('%', '')
            try:
                value = float(text) / 100
                values.append(value)
            except ValueError:
                raise ValueError(f"Invalid percentage value in {row_name}, year {col-1}")
        return values
    
    def parse_float_row(self, row_name, include_terminal=False):
        """Parse a row of float values from the table"""
        row = self.find_row(row_name)
        values = []
        end_col = 12 if include_terminal else 11
        for col in range(2, end_col):  # Years 1-10 (or + terminal if specified)
            text = self.table.item(row, col).text()
            try:
                value = float(text)
                values.append(value)
            except ValueError:
                raise ValueError(f"Invalid numeric value in {row_name}, year {col-1}")
        return values
    
    def find_row(self, row_name):
        """Find the row index for a given row name"""
        for row in range(self.table.rowCount()):
            if self.table.verticalHeaderItem(row).text() == row_name:
                return row
        return -1
    
    def update_calculated_values(self, forecast_df, roic_df, valuation):
        """Update the table with calculated values from the FCFF functions"""
        # Helper function to update a row from forecast_df
        def update_row_from_forecast(row_name, df_name=None, format_str="{:,.2f}"):
            df_name = df_name or row_name
            row = self.find_row(row_name)
            if row >= 0 and df_name in forecast_df.index:
                for col in range(2, 12):  # Years 1-10
                    year = f"Year {col-1}"
                    if year in forecast_df.columns:
                        value = forecast_df.loc[df_name, year]
                        if isinstance(value, (int, float)):
                            self.table.item(row, col).setText(format_str.format(value))
        
        # Update Revenue
        update_row_from_forecast("Revenue")
        
        # Update EBIT
        update_row_from_forecast("EBIT")
        
        # Update EBT after Tax
        update_row_from_forecast("EBT after Tax", "EBIT after Tax")
        
        # Update Reinvestment
        update_row_from_forecast("Reinvestment")
        
        # Update FCFF
        update_row_from_forecast("FCFF")
        
        # Update Discount Factor
        update_row_from_forecast("Discount Factor", format_str="{:.4f}")
        
        # Update Discounted FCFF
        update_row_from_forecast("Discounted FCFF")
        
        # Update Sum of Discounted FCFF
        sum_row = self.find_row("Sum of Discounted FCFF")
        if sum_row >= 0 and "Sum of Discounted FCFF" in forecast_df.index:
            self.table.item(sum_row, 1).setText("{:,.2f}".format(forecast_df.loc["Sum of Discounted FCFF", "Value"]))
        
        # Update Terminal Value components
        tv_fcff_row = self.find_row("TV-FCFF")
        terminal_value_row = self.find_row("Terminal Value = TV-FCFF/(k-g)")
        discounted_terminal_row = self.find_row("Discounted Terminal Value")
        
        if all(row >= 0 for row in [tv_fcff_row, terminal_value_row, discounted_terminal_row]):
            if "TV-FCFF" in forecast_df.index:
                self.table.item(tv_fcff_row, 12).setText("{:,.2f}".format(forecast_df.loc["TV-FCFF", "Terminal Value"]))
            if "Terminal Value" in forecast_df.index:
                self.table.item(terminal_value_row, 12).setText("{:,.2f}".format(forecast_df.loc["Terminal Value", "Terminal Value"]))
            if "Discounted Terminal Value" in forecast_df.index:
                self.table.item(discounted_terminal_row, 12).setText("{:,.2f}".format(forecast_df.loc["Discounted Terminal Value", "Terminal Value"]))
        
        # Update Total Firm Value
        firm_value_row = self.find_row("Total Firm Value")
        if firm_value_row >= 0 and "Total Firm Value" in forecast_df.index:
            self.table.item(firm_value_row, 1).setText("{:,.2f}".format(forecast_df.loc["Total Firm Value", "Value"]))
        
        # Update Equity Value and Fair Value per Share
        equity_row = self.find_row("Equity Value")
        shares_row = self.find_row("Shares Outstanding")
        fair_value_row = self.find_row("Fair Value per Share")
        
        if all(row >= 0 for row in [equity_row, shares_row, fair_value_row]):
            if "Equity Value" in forecast_df.index:
                self.table.item(equity_row, 1).setText("{:,.2f}".format(forecast_df.loc["Equity Value", "Value"]))
            if "Shares Outstanding" in forecast_df.index:
                shares = float(self.table.item(shares_row, 1).text().replace(',', ''))
                equity_value = float(forecast_df.loc['Equity Value', 'Value'].replace(',', ''))
                fair_value = equity_value / shares
                self.table.item(fair_value_row, 1).setText("{:,.2f}".format(fair_value))
        
        # Update ROIC-related fields
        invested_capital_row = self.find_row("Invested Capital")
        avg_invested_row = self.find_row("Avg Invested Capital")
        roic_row = self.find_row("Return on Invested Capital")
        
        if all(row >= 0 for row in [invested_capital_row, avg_invested_row, roic_row]):
            for col in range(2, 12):  # Years 1-10
                year = f"Year {col-1}"
                if year in roic_df.columns:
                    if "Invested Capital" in roic_df.index:
                        self.table.item(invested_capital_row, col).setText("{:,.2f}".format(roic_df.loc["Invested Capital", year]))
                    if "Avg Invested Capital" in roic_df.index:
                        self.table.item(avg_invested_row, col).setText("{:,.2f}".format(roic_df.loc["Avg Invested Capital", year]))
                    if "ROIC" in roic_df.index:
                        self.table.item(roic_row, col).setText("{:.2%}".format(roic_df.loc["ROIC", year]))
            
            # Set terminal ROIC (after 10 years)
            self.table.item(roic_row, 12).setText("{:.2%}".format(roic_df.loc["Terminal ROIC", "Terminal Value"]))

            
class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

class HistoricalDataTab(QWidget):
    """Tab 2: Historical Data"""
    def __init__(self, parent=None):
        super(HistoricalDataTab, self).__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout()
        
        # Header that will update based on selected stock
        self.header = QLabel("Historical Financial Overview")
        self.header.setFont(QFont('Arial', 14, QFont.Bold))
        self.header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.header)
        
        # Create a scroll area for the grid of plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        
        scroll.setWidget(self.scroll_content)
        self.layout.addWidget(scroll)
        self.setLayout(self.layout)
        
        # Create a folder for saving plots
        self.plots_dir = "saved_plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Store current company to detect changes
        self.current_company = None
    
    def update_plots(self, fcff):
        """Update the plots with data for the selected stock"""
        try:
            # Update header
            company = "Microsoft" if fcff.stock == "MSFT" else "NVIDIA"
            self.header.setText(f"Historical Financial Overview - {company}")
            
            # Clear previous plots from grid
            self._clear_grid_layout()
            
            # Always regenerate plots when company changes
            if company != self.current_company:
                self.current_company = company
                plot_files = self._generate_and_save_plots(fcff, company)
                self._arrange_plots_in_grid(plot_files)
            
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    def _clear_grid_layout(self):
        """Clear all widgets from the grid layout"""
        for i in reversed(range(self.grid_layout.count())): 
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
    
    def _generate_and_save_plots(self, fcff, company):
        """Generate all plots and save them to files (always overwrites)"""
        plot_files = {}
        
        # Generate and save each plot (always overwrite)
        fig, ax = fcff.plot_reinvestment_only()
        reinvestment_path = self._save_plot(fig, f"{company}_reinvestment.png")
        plt.close(fig)
        
        fig, ax = fcff.plot_revenue_and_growth()
        revenue_path = self._save_plot(fig, f"{company}_revenue.png")
        plt.close(fig)
        
        fig, ax = fcff.plot_ebit()
        ebit_path = self._save_plot(fig, f"{company}_ebit.png")
        plt.close(fig)
        
        fig, ax1, ax2 = fcff.plot_invested_capital_and_roic()
        invested_capital_path = self._save_plot(fig, f"{company}_invested_capital.png")
        plt.close(fig)
        
        fig, ax = fcff.plot_operating_margin()
        operating_margin_path = self._save_plot(fig, f"{company}_operating_margin.png")
        plt.close(fig)
        
        fig, ax = fcff.plot_stock_price()
        stock_price_path = self._save_plot(fig, f"{company}_stock_price.png")
        plt.close(fig)
        
        # Return paths to all generated plots
        return {
            'reinvestment': reinvestment_path,
            'revenue': revenue_path,
            'ebit': ebit_path,
            'invested_capital': invested_capital_path,
            'operating_margin': operating_margin_path,
            'stock_price': stock_price_path
        }
    
    def _save_plot(self, fig, filename):
        """Save a matplotlib figure to file (always overwrites)"""
        filepath = os.path.join(self.plots_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=100)
        return filepath
    
    def _arrange_plots_in_grid(self, plot_files):
        """Arrange the saved plots in the grid layout according to specifications"""
        # Column 1: reinvestment, revenue, ebit (stacked vertically)
        self._add_plot_to_grid(plot_files['reinvestment'], 0, 1, "Reinvestment")
        self._add_plot_to_grid(plot_files['revenue'], 5, 1, "Revenue and Growth")
        self._add_plot_to_grid(plot_files['ebit'], 10, 1, "EBIT")
        
        # Column 2: invested capital/roic and operating margin (offset vertically)
        self._add_plot_to_grid(plot_files['invested_capital'], 2, 7, "Invested Capital and ROIC")
        self._add_plot_to_grid(plot_files['operating_margin'], 7, 7, "Operating Margin")
        
        # Column 3: stock price (centered vertically)
        self._add_plot_to_grid(plot_files['stock_price'], 5, 13, "Stock Price")
    
    def _add_plot_to_grid(self, filepath, row, col, title, rowspan=5):
        """Add a saved plot image to the grid layout"""
        canvas = MplCanvas(self, width=5, height=4)
        
        # Clear any existing content
        canvas.axes.clear()
        
        # Display the saved image
        img = plt.imread(filepath)
        canvas.axes.imshow(img)
        canvas.axes.axis('off')  # Hide axes
        canvas.axes.set_title(title)
        
        # Add to grid layout
        self.grid_layout.addWidget(canvas, row, col, rowspan, 4)
        
        # Redraw the canvas
        canvas.draw()

class RevenueBySegmentTab(QWidget):
    """Tab 3: Responsive Revenue by Segment with embedded Sankey diagrams"""
    def __init__(self, parent=None):
        super(RevenueBySegmentTab, self).__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)  # Set layout directly to self
        self.layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for full-width display
        
        # Header that will update based on selected stock
        self.header = QLabel("Revenue by Segment")
        self.header.setFont(QFont('Arial', 14, QFont.Bold))
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout.addWidget(self.header, stretch=0)  # Header won't stretch
        
        # Create web view container with proper size policies
        self.web_container = QWidget()
        self.web_layout = QVBoxLayout(self.web_container)
        self.web_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create web view for Sankey diagram
        self.web_view = QWebEngineView()
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add web view to container
        self.web_layout.addWidget(self.web_view)
        
        # Add container to main layout with stretch factor
        self.layout.addWidget(self.web_container, stretch=1)  # Will expand to fill space
        
        # Temporary HTML file paths
        self.msft_html_path = os.path.abspath("msftsankey.html")
        self.nvda_html_path = os.path.abspath("nvda_sankey_final.html")
        
        # Connect resize event
        self.web_view.loadFinished.connect(self.adjust_web_view_size)
    
    def adjust_web_view_size(self, *args):
        """Adjust web view size after content loads"""
        # Set zoom factor based on container size
        container_width = self.web_container.width()
        if container_width > 0:
            # Adjust zoom to maintain reasonable diagram size
            zoom_factor = min(1.0, max(0.7, container_width / 1200))
            self.web_view.setZoomFactor(zoom_factor)
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if self.web_view.url().isEmpty():
            return
        self.adjust_web_view_size()
    
    def update_plot(self, fcff):
        """Update the Sankey diagram based on selected stock"""
        try:
            # Generate the appropriate Sankey diagram
            if fcff.stock == "MSFT":
                self.header.setText("Microsoft FY2024 Segment Breakdown")
                fcff.sankey_microsoft()
                html_path = self.msft_html_path
            else:
                self.header.setText("NVIDIA FY2025 Segment Breakdown")
                fcff.sankey_nvidia()
                html_path = self.nvda_html_path
            
            # Load the generated HTML file into the web view
            if os.path.exists(html_path):
                self.web_view.setUrl(QUrl.fromLocalFile(html_path))
                
            else:
                raise FileNotFoundError(f"HTML file not found: {html_path}")
            
        except Exception as e:
            print(f"Error updating Sankey diagram: {e}")
            # Fallback to responsive error message
            self.web_view.setHtml("""
                <html>
                    <head>
                        <style>
                            body {
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                height: 100vh;
                                margin: 0;
                                font-family: Arial, sans-serif;
                            }
                            .error-container {
                                text-align: center;
                                max-width: 80%;
                            }
                            h1 { color: #666; font-size: 1.5vw; }
                            p { color: #999; font-size: 1vw; }
                        </style>
                    </head>
                    <body>
                        <div class="error-container">
                            <h1>Error loading Sankey diagram</h1>
                            <p>Could not generate the revenue segment visualization</p>
                        </div>
                    </body>
                </html>
            """)

class MonteCarloTab(QWidget):
    """Tab 4: Monte Carlo Simulation with image-based results"""
    def __init__(self, parent=None):
        super(MonteCarloTab, self).__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout()
        
        # Create directory for saving plots if it doesn't exist
        self.plot_dir = "monte_carlo_plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Header
        header = QLabel("Monte Carlo Simulation - NVIDIA")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(header)
        
        # Instruction
        instruction = QLabel("Enter the number of simulations and press the button below to perform a valuation of NVIDIA using a Monte Carlo approach.")
        instruction.setWordWrap(True)
        self.layout.addWidget(instruction)
        
        # Input for number of simulations
        sim_layout = QHBoxLayout()
        sim_layout.addWidget(QLabel("Number of simulations:"))
        self.sim_input = QLineEdit("1000")
        self.sim_input.setValidator(QIntValidator(1, 1000000))  # Only allow positive integers
        sim_layout.addWidget(self.sim_input)
        sim_layout.addStretch()
        self.layout.addLayout(sim_layout)
        
        # Run button
        self.run_btn = QPushButton("Run Monte Carlo Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        self.layout.addWidget(self.run_btn)
        
        # Results area with tabs for plots
        self.results_tabs = QTabWidget()
        
        # Distribution plot tab
        self.dist_tab = QWidget()
        self.dist_layout = QVBoxLayout()
        self.dist_image_label = QLabel()
        self.dist_image_label.setAlignment(Qt.AlignCenter)
        self.dist_scroll = QScrollArea()
        self.dist_scroll.setWidgetResizable(True)
        self.dist_scroll.setWidget(self.dist_image_label)
        self.dist_layout.addWidget(self.dist_scroll)
        self.dist_tab.setLayout(self.dist_layout)
        self.results_tabs.addTab(self.dist_tab, "Distribution")
        
        # Percentile table tab
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout()
        self.table_image_label = QLabel()
        self.table_image_label.setAlignment(Qt.AlignCenter)
        self.table_scroll = QScrollArea()
        self.table_scroll.setWidgetResizable(True)
        self.table_scroll.setWidget(self.table_image_label)
        self.table_layout.addWidget(self.table_scroll)
        self.table_tab.setLayout(self.table_layout)
        self.results_tabs.addTab(self.table_tab, "Percentile Table")
        
        self.layout.addWidget(self.results_tabs)
        self.setLayout(self.layout)
    
    def run_simulation(self):
        """Run the Monte Carlo simulation and display results as images"""
        try:
            num_simulations = int(self.sim_input.text())
            if num_simulations <= 0:
                raise ValueError("Number of simulations must be positive")
            
            # Run simulation
            mc = MonteCarloInputSimulator(num_simulations)
            
            # Generate and save distribution plot
            dist_fig, _ = mc.plot_fair_value_distribution(
                mc.fair_values,
                stock_name='NVDA',
                stock_price=mc.fcff.bloomberg_data["stock_price"]
            )
            dist_path = os.path.join(self.plot_dir, "distribution.png")
            dist_fig.savefig(dist_path, bbox_inches='tight', dpi=150)
            plt.close(dist_fig)
            
            # Generate and save percentile table
            table_fig, _ = mc.plot_percentile_table(mc.fair_values)
            table_path = os.path.join(self.plot_dir, "percentile_table.png")
            table_fig.savefig(table_path, bbox_inches='tight', dpi=150)
            plt.close(table_fig)
            
            # Display the saved images
            self.display_plot_image(dist_path, self.dist_image_label)
            self.display_plot_image(table_path, self.table_image_label)
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid number of simulations: {str(e)}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to run simulation: {str(e)}")
            # Clear displays on error
            self.dist_image_label.clear()
            self.table_image_label.clear()
    
    def display_plot_image(self, image_path, label):
        """Display an image from file in a QLabel with proper scaling"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale pixmap to fit the label while maintaining aspect ratio
                label.setPixmap(pixmap.scaled(
                    label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
            else:
                label.setText("Error: Could not load result image")
        else:
            label.setText("Error: Result image not found")
    
    def resizeEvent(self, event):
        """Handle window resize to properly scale images"""
        super().resizeEvent(event)
        # Update image scaling when window is resized
        dist_path = os.path.join(self.plot_dir, "distribution.png")
        table_path = os.path.join(self.plot_dir, "percentile_table.png")
        
        if os.path.exists(dist_path):
            self.display_plot_image(dist_path, self.dist_image_label)
        if os.path.exists(table_path):
            self.display_plot_image(table_path, self.table_image_label)

class StoryTab(QWidget):
    """Tab 5: Story Behind the Numbers with 2x2 grid layout"""
    def __init__(self, parent=None):
        super(StoryTab, self).__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout()
        
        # Create directory for saving plots if it doesn't exist
        self.plot_dir = "story_plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Header
        header = QLabel("Story Behind the Numbers")
        header.setFont(QFont('Arial', 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(header)
        
        # Create a scroll area for the content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        
        # Create a grid layout for the distribution blocks (2x2)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(20)  # Add spacing between blocks
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create 4 distribution blocks
        self.distribution_blocks = [
            {
                "title": "Revenue Growth",
                "type": "normal",
                "params": {"mean": 0.30, "std": 0.02},
                "description": "Normal Distribution (mean = 0.30, std = 0.02)"
            },
            {
                "title": "Operating Margin",
                "type": "normal",
                "params": {"mean": 0.45, "std": 0.05},
                "description": "Normal Distribution (mean = 0.45, std = 0.05)"
            },
            {
                "title": "Tax Rate",
                "type": "triangular",
                "params": {"min": 0.10, "mode": 0.15, "max": 0.20},
                "description": "Triangular Distribution (min = 0.10, mode = 0.15, max = 0.20)"
            },
            {
                "title": "WACC",
                "type": "lognormal",
                "params": {"mean": 0.08, "std": 0.01},
                "description": "Lognormal Distribution (mean = 0.08, std = 0.01)"
            }
        ]
        
        # Add blocks to grid layout
        for i, block_info in enumerate(self.distribution_blocks):
            row = i // 2
            col = i % 2
            self.create_distribution_block(block_info, row, col)
        
        # Add grid layout to scroll content
        self.scroll_layout.addLayout(self.grid_layout)
        self.scroll_layout.addStretch()  # Add stretch to push content up
        
        scroll.setWidget(scroll_content)
        self.layout.addWidget(scroll)
        self.setLayout(self.layout)
    
    def create_distribution_block(self, block_info, row, col):
        """Create a distribution block and add it to the grid"""
        block = QGroupBox(block_info["title"])
        block_layout = QVBoxLayout()  # Changed to vertical layout
        
        # Plot image
        plot_label = QLabel()
        plot_label.setAlignment(Qt.AlignCenter)
        plot_label.setMinimumSize(300, 200)  # Smaller size for grid layout
        
        # Generate and save plot image
        img_path = os.path.join(self.plot_dir, f"{block_info['title'].lower().replace(' ', '_')}.png")
        self.generate_distribution_plot(
            block_info["type"],
            block_info["params"],
            block_info["title"],
            img_path
        )
        
        # Display the saved image
        self.display_plot_image(img_path, plot_label)
        block_layout.addWidget(plot_label, stretch=2)  # Plot takes more space
        
        # Description label
        desc_label = QLabel(block_info["description"])
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        block_layout.addWidget(desc_label, stretch=0)
        
        # Explanation text
        text_box = QTextEdit()
        text_box.setPlainText(self.get_distribution_description(block_info))
        text_box.setReadOnly(True)
        text_box.setMaximumHeight(80)
        text_box.setStyleSheet("QTextEdit { background-color: #f8f9fa; }")
        block_layout.addWidget(text_box, stretch=1)
        
        block.setLayout(block_layout)
        self.grid_layout.addWidget(block, row, col)
    
    def generate_distribution_plot(self, dist_type, params, title, save_path):
        """Generate and save a distribution plot (same as before)"""
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)  # Smaller figure size
        
        if dist_type == "normal":
            x = np.linspace(params["mean"] - 3*params["std"], 
                           params["mean"] + 3*params["std"], 100)
            y = (1/(params["std"] * np.sqrt(2*np.pi))) * \
                np.exp(-0.5*((x-params["mean"])/params["std"])**2)
            ax.plot(x, y, color='#2077b4', linewidth=1.5)
            ax.fill_between(x, y, color='#2077b4', alpha=0.2)
            
        elif dist_type == "triangular":
            from scipy.stats import triang
            c = (params["mode"] - params["min"]) / (params["max"] - params["min"])
            x = np.linspace(params["min"], params["max"], 100)
            y = triang.pdf(x, c, loc=params["min"], scale=params["max"]-params["min"])
            ax.plot(x, y, color='#2bdab3', linewidth=1.5)
            ax.fill_between(x, y, color='#2bdab3', alpha=0.2)
            
        elif dist_type == "lognormal":
            from scipy.stats import lognorm
            sigma = params["std"]
            mean = params["mean"]
            x = np.linspace(0, mean + 4*sigma, 100)
            y = lognorm.pdf(x, sigma, scale=np.exp(mean))
            ax.plot(x, y, color='#e27373', linewidth=1.5)
            ax.fill_between(x, y, color='#e27373', alpha=0.2)
        
        # Format plot
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save plot
        fig.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
    
    def display_plot_image(self, image_path, label):
        """Display an image from file in a QLabel with proper scaling"""
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale pixmap to fit the label while maintaining aspect ratio
                label.setPixmap(pixmap.scaled(
                    label.width(), label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
    
    def get_distribution_description(self, block_info):
        """Get explanatory text for each distribution type (same as before)"""
        if block_info["type"] == "normal":
            return (f"The {block_info['title']} follows a normal distribution, which assumes symmetric "
                   f"variation around the mean value of {block_info['params']['mean']:.2f}. This is "
                   f"appropriate for metrics that can vary both positively and negatively with "
                   f"equal probability.")
        
        elif block_info["type"] == "triangular":
            return (f"The {block_info['title']} uses a triangular distribution bounded between "
                   f"{block_info['params']['min']:.2f} and {block_info['params']['max']:.2f}, with "
                   f"the most likely value at {block_info['params']['mode']:.2f}. This captures "
                   f"asymmetric uncertainty where extreme values are less probable.")
        
        elif block_info["type"] == "lognormal":
            return (f"The {block_info['title']} is modeled with a lognormal distribution, which is "
                   f"appropriate for values that are always positive and may have a long tail. "
                   f"The distribution has a mean of {block_info['params']['mean']:.2f} with "
                   f"standard deviation {block_info['params']['std']:.2f}.")
        
        return "Description of the distribution's impact on valuation."
    
    def resizeEvent(self, event):
        """Handle window resize to properly scale images"""
        super().resizeEvent(event)
        # Update all plot images when window is resized
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if isinstance(widget, QGroupBox):
                plot_label = widget.layout().itemAt(0).widget()
                if isinstance(plot_label, QLabel):
                    block_info = self.distribution_blocks[i]
                    img_path = os.path.join(self.plot_dir, f"{block_info['title'].lower().replace(' ', '_')}.png")
                    if os.path.exists(img_path):
                        self.display_plot_image(img_path, plot_label)

class StockValuationDashboard(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Stock Valuation Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize with Microsoft as default
        self.stock = "MSFT"
        self.fcff = FCFFModel(stock=self.stock)
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Add tabs
        self.reverse_fcff_tab = ReverseFCFFTab(self)
        self.historical_data_tab = HistoricalDataTab(self)
        self.revenue_segment_tab = RevenueBySegmentTab(self)
        self.monte_carlo_tab = MonteCarloTab(self)
        self.story_tab = StoryTab(self)
        
        self.tabs.addTab(self.reverse_fcff_tab, "Reverse FCFF Tool")
        self.tabs.addTab(self.historical_data_tab, "Historical Data")
        self.tabs.addTab(self.revenue_segment_tab, "Revenue by Segment")
        self.tabs.addTab(self.monte_carlo_tab, "Monte Carlo Simulation")
        self.tabs.addTab(self.story_tab, "Story Behind the Numbers")
        
        # Set the first tab as the central widget
        self.setCentralWidget(self.tabs)
        
        # Update all tabs with initial data
        self.update_all_tabs()
    
    def update_stock(self, stock):
        """Update the current stock and refresh all tabs"""
        if stock != self.stock:
            self.stock = stock
            self.fcff = FCFFModel(stock=self.stock)
            self.update_all_tabs()
    
    def update_all_tabs(self):
        """Update all tabs with current stock data"""
        self.historical_data_tab.update_plots(self.fcff)
        self.revenue_segment_tab.update_plot(self.fcff)
        
        # Disable Monte Carlo tab if not NVIDIA
        self.tabs.setTabEnabled(3, self.stock == "NVDA")
    
    def update_calculations(self, forecast_df, valuation, roic_df):
        """Update calculations that might be used by other tabs"""
        # Currently just storing, could be used to update other tabs
        self.forecast_df = forecast_df
        self.valuation = valuation
        self.roic_df = roic_df


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockValuationDashboard()
    window.show()
    sys.exit(app.exec_())