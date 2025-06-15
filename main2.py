# from curses import COLORS
import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define color palette for consistent styling
COLORS = {
    'primary': '#2077b4',
    'secondary': '#2bdab3',
    'danger': '#e27373',
    'warning': '#f7b731',
    'success': '#27ae60',
    'info': '#2980b9',
    'background': '#f8f9fa'
}

from PyQt5.QtCore import Qt, QUrl, QSize
from PyQt5.QtGui import QFont, QPixmap, QIntValidator, QColor, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QLineEdit, QTextEdit, QGroupBox, QScrollArea, QMessageBox, QSizePolicy, QGridLayout, QFrame
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

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, QTableWidget, 
                            QTableWidgetItem, QPushButton, QHBoxLayout, QHeaderView,
                            QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor
from FCFF import FCFFModel

class ReverseFCFFTab(QWidget):
    """Tab 1: Reverse FCFF Tool with complete valuation table"""
    def __init__(self, parent=None):
        super(ReverseFCFFTab, self).__init__(parent)
        self.parent = parent
        self.user_inputs = {"MSFT": None, "NVDA": None}  # Store inputs per company
        self.current_stock = None
        self.layout = QVBoxLayout(self)
        self.setup_ui()
        self.update_company()
        self.calculate_fair_value()
    
    def setup_ui(self):
        self.setup_styles()
        self.setup_header()
        self.setup_company_selection()
        self.setup_instruction_label()
        self.create_valuation_table()
        self.setup_calculate_button()
        self.setup_value_display()
    
    def setup_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
            QTableWidget {
                border: 1px solid #d3d3d3;
                border-radius: 4px;
                gridline-color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 5px;
                border: none;
                font-weight: bold;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QComboBox {
                padding: 3px;
                border: 1px solid #d3d3d3;
                border-radius: 4px;
                min-height: 25px;
            }
            QLabel#valueDisplay {
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 10px;
            }
        """)
    
    def setup_header(self):
        # Modern header with company name
        self.header = QLabel("Reverse FCFF Tool")
        self.header.setFont(QFont('Segoe UI', 20, QFont.Bold))
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(f"""
            color: {COLORS['primary']};
            padding: 12px;
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 8px;
        """)
        self.layout.addWidget(self.header)
    
    def setup_company_selection(self):
        self.company_combo = QComboBox()
        self.company_combo.addItems(["Microsoft", "NVIDIA"])
        self.company_combo.currentTextChanged.connect(self.update_company)
        
        company_layout = QHBoxLayout()
        company_layout.addWidget(QLabel("Select Company:"))
        company_layout.addWidget(self.company_combo)
        company_layout.addStretch()
        self.layout.addLayout(company_layout)
    
    def setup_instruction_label(self):
        instruction = QLabel("Please enter your valuation assumptions below and click the button to calculate the fair value.")
        instruction.setWordWrap(True)
        instruction.setStyleSheet("color: #7f8c8d; margin-bottom: 10px;")
        self.layout.addWidget(instruction)
    
    def create_valuation_table(self):
        self.table = QTableWidget()
        self.table.setColumnCount(12)  # Base year + 10 years + terminal
        self.table.setRowCount(23)
        
        headers = ["Base Year"] + [f"Year {i+1}" for i in range(10)] + ["Terminal"]
        self.table.setHorizontalHeaderLabels(headers)
        
        row_labels = [
            "Revenue Growth", "Revenue", "Operating Margin", "EBIT", "Tax Rate",
            "EBT after Tax", "Reinvestment Rate", "Reinvestment", "FCFF", "WACC",
            "Discount Factor", "Discounted FCFF", "Sum of Discounted FCFF",
            "TV-FCFF", "Terminal Value", "Discounted Terminal Value",
            "Total Firm Value", "Total Debt", "plus Cash", "Equity Value",
            "Shares Outstanding", "Fair Value per Share", "Return on Invested Capital"
        ]
        self.table.setVerticalHeaderLabels(row_labels)
        
        # Configure responsive table sizing
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # Set editable cells
        editable_rows = {
            "Revenue Growth": range(0, 12),
            "Operating Margin": range(0, 12),
            "Tax Rate": range(0, 12),
            "Reinvestment Rate": range(0, 11),
            "WACC": range(0, 12)
        }
        
        for row in range(self.table.rowCount()):
            row_name = row_labels[row]
            for col in range(self.table.columnCount()):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                
                if row_name in editable_rows and col in editable_rows[row_name]:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    item.setBackground(QColor(255, 229, 204))  # Light orange
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    item.setBackground(QColor(240, 240, 240))
                
                self.table.setItem(row, col, item)
        
        self.layout.addWidget(self.table)
    
    def setup_calculate_button(self):
        self.calculate_btn = QPushButton("Calculate Fair Value")
        self.calculate_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.calculate_btn.clicked.connect(self.calculate_fair_value)
        
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addStretch()
        btn_layout.addWidget(self.calculate_btn)
        btn_layout.addStretch()
        
        self.layout.addWidget(btn_container)
    
    def setup_value_display(self):
        value_container = QWidget()
        value_container.setObjectName("valueDisplay")
        value_layout = QVBoxLayout(value_container)
        
        # Company display
        self.company_display = QLabel()
        self.company_display.setAlignment(Qt.AlignCenter)
        self.company_display.setStyleSheet("""
            font-weight: bold; 
            font-size: 12px; 
            color: #2c3e50;
            margin-bottom: 8px;
        """)
        value_layout.addWidget(self.company_display)
        
        # Value display
        value_display_layout = QHBoxLayout()
        
        self.fair_value_label = QLabel("Fair Value per Share: -")
        self.fair_value_label.setStyleSheet("""
            font-weight: bold; 
            color: #27ae60;
            font-size: 11pt;
        """)
        
        self.market_price_label = QLabel("Market Price: -")
        self.market_price_label.setStyleSheet("""
            font-weight: bold; 
            color: #e74c3c;
            font-size: 11pt;
        """)
        
        value_display_layout.addStretch()
        value_display_layout.addWidget(self.fair_value_label)
        value_display_layout.addSpacing(20)
        value_display_layout.addWidget(self.market_price_label)
        value_display_layout.addStretch()
        
        value_layout.addLayout(value_display_layout)
        self.layout.addWidget(value_container)
    
    def update_company(self):
        company = self.company_combo.currentText()
        stock = "MSFT" if company == "Microsoft" else "NVDA"
        self.current_stock = stock
        
        # Save current inputs before switching
        if hasattr(self, 'fcff') and self.fcff and self.current_stock:
            self.save_current_inputs()
        
        self.company_display.setText(f"Current Selection: {company} ({stock})")
        self.fcff = FCFFModel(stock=stock)
        
        # Update market price display
        market_price = self.fcff.bloomberg_data["stock_price"]
        reporting_date = self.fcff.bloomberg_data["reporting_date"]
        self.market_price_label.setText(f"Market Price: {market_price:.4f} ({reporting_date:%d.%m.%Y})")
        
        # Load saved inputs or defaults
        self.load_inputs(stock)
        self.set_base_year_values()
        self.parent.update_stock(stock)
        self.calculate_fair_value()
    
    def save_current_inputs(self):
        """Save current inputs to dictionary before switching companies"""
        try:
            inputs = {
                "revenue_growth": self.parse_percentage_row("Revenue Growth", include_terminal=True),
                "operating_margin": self.parse_percentage_row("Operating Margin", include_terminal=True),
                "tax_rate": self.parse_percentage_row("Tax Rate", include_terminal=True),
                "reinvestment_rate": self.parse_float_row("Reinvestment Rate"),
                "wacc": self.parse_float_row("WACC", include_terminal=True),
                "roic_tv": float(self.table.item(self.find_row("Return on Invested Capital"), 11).text())
            }
            self.user_inputs[self.current_stock] = inputs
        except Exception as e:
            print(f"Error saving inputs: {str(e)}")
    
    def load_inputs(self, stock):
        """Load saved inputs or defaults for the selected stock"""
        if self.user_inputs[stock] is not None:
            # Load saved inputs
            inputs = self.user_inputs[stock]
            self.set_input_values(inputs)
        else:
            # Load default assumptions
            self.load_default_assumptions(stock)
    
    def set_input_values(self, inputs):
        """Set input values from dictionary to table"""
        row_mapping = {
            "revenue_growth": "Revenue Growth",
            "operating_margin": "Operating Margin",
            "tax_rate": "Tax Rate",
            "reinvestment_rate": "Reinvestment Rate",
            "wacc": "WACC"
        }
        
        # Find row indices
        row_indices = {}
        for row in range(self.table.rowCount()):
            row_name = self.table.verticalHeaderItem(row).text()
            for param, label in row_mapping.items():
                if row_name == label:
                    row_indices[param] = row
                    break
        
        # Set values from inputs
        for param, row in row_indices.items():
            values = inputs[param]
            for col in range(1, len(values) + 1):  # Start from Year 1
                if param == "reinvestment_rate" and col == 11:  # Skip terminal for reinvestment
                    continue
                self.table.item(row, col).setText(f"{values[col-1]:.4f}")
        
        # Set terminal ROIC
        roic_row = self.find_row("Return on Invested Capital")
        if roic_row >= 0:
            self.table.item(roic_row, 11).setText(f"{inputs['roic_tv']:.4f}")
    
    def load_default_assumptions(self, stock):
        if stock == "NVDA":
            defaults = {
                "revenue_growth": [0.10, 0.10, 0.10, 0.10, 0.10, 0.0986, 0.0972, 0.0858, 0.0644, 0.043, 0.043],
                "operating_margin": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                "tax_rate": [0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142],
                "reinvestment_rate": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "wacc": [0.098, 0.098, 0.098, 0.098, 0.098, 0.0952, 0.0924, 0.0896, 0.0868, 0.0840, 0.0840],
                "roic_tv": 0.2
            }
        else:
            defaults = {
                "revenue_growth": [0.15, 0.15, 0.15, 0.15, 0.15, 0.1286, 0.1072, 0.0858, 0.0644, 0.043, 0.043],
                "operating_margin": [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
                "tax_rate": [0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182, 0.182],
                "reinvestment_rate": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                "wacc": [0.098, 0.098, 0.098, 0.098, 0.098, 0.0952, 0.0924, 0.0896, 0.0868, 0.0840, 0.0840],
                "roic_tv": 0.2
            }
        
        self.set_input_values(defaults)
    
    def set_base_year_values(self):
        data = self.fcff.bloomberg_data
        base_values = {
            "Revenue": data["base_year_revenue"],
            "Operating Margin": data["ebit_balance"] / data["base_year_revenue"],
            "EBIT": data["ebit_balance"],
            "Tax Rate": data["base_year_tax"],
            "Total Debt": data["Total debt"],
            "plus Cash": data["Cash"],
            "Shares Outstanding": data["Shares outstanding"]
        }
        
        # Set base year values (column 0)
        for row in range(self.table.rowCount()):
            row_name = self.table.verticalHeaderItem(row).text()
            if row_name == "Revenue":
                self.table.item(row, 0).setText(f"{base_values['Revenue']:,.4f}")
            elif row_name == "Operating Margin":
                self.table.item(row, 0).setText(f"{base_values['Operating Margin']:.4f}")
            elif row_name == "EBIT":
                self.table.item(row, 0).setText(f"{base_values['EBIT']:,.4f}")
            elif row_name == "Tax Rate":
                self.table.item(row, 0).setText(f"{base_values['Tax Rate']:.4f}")
            elif row_name == "Total Debt":
                self.table.item(row, 0).setText(f"{base_values['Total Debt']:,.4f}")
            elif row_name == "plus Cash":
                self.table.item(row, 0).setText(f"{base_values['plus Cash']:,.4f}")
            elif row_name == "Shares Outstanding":
                self.table.item(row, 0).setText(f"{base_values['Shares Outstanding']:,.4f}")
    
    def calculate_fair_value(self):
        try:
            # Save current inputs
            self.save_current_inputs()
            
            # Get current inputs
            user_input = self.user_inputs[self.current_stock]
            
            # Run calculations
            _, forecast_df = self.fcff.build_forecast_df(user_inputs=user_input)
            valuation = self.fcff.calculate_valuation(user_inputs=user_input)
            _, roic_df = self.fcff.build_roic_df(user_inputs=user_input)
            
            # Update table with results
            self.update_table_with_results(forecast_df, valuation, roic_df, user_input)
            
            # Update fair value display
            fair_value = valuation.loc['Fair Value per Share', 'Value']
            self.fair_value_label.setText(f"Fair Value per Share: {fair_value:,.4f}")
            
            # Update parent
            self.parent.update_calculations(forecast_df, valuation, roic_df)
            
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", f"Error during calculation: {str(e)}")
    
    def update_table_with_results(self, forecast_df, valuation, roic_df, user_input):
        # Update forecast years (columns 1-10)
        for year in range(10):
            year_col = year + 1
            year_key = forecast_df.columns[year]
            
            # Update financial metrics
            self.update_table_cell("Revenue", year_col, forecast_df.loc['Revenue', year_key], ",.4f")
            self.update_table_cell("EBIT", year_col, forecast_df.loc['EBIT', year_key], ",.4f")
            self.update_table_cell("EBT after Tax", year_col, forecast_df.loc['EBIT after Tax', year_key], ",.4f")
            self.update_table_cell("Reinvestment", year_col, forecast_df.loc['Reinvestment', year_key], ",.4f")
            self.update_table_cell("FCFF", year_col, forecast_df.loc['FCFF', year_key], ",.4f")
            self.update_table_cell("Discount Factor", year_col, forecast_df.loc['Discount Factor', year_key], ".4f")
            self.update_table_cell("Discounted FCFF", year_col, forecast_df.loc['Discounted FCFF', year_key], ",.4f")
            
            # Add ROIC if exists in dataframe
            if 'Return on Invested Capital' in roic_df.index:
                self.update_table_cell("Return on Invested Capital", year_col, 
                                     roic_df.loc['Return on Invested Capital', year_key], ".4f")
        
        # Update terminal values (column 11)
        terminal_col = 11
        self.update_table_cell("TV-FCFF", terminal_col, valuation.loc['TV - FCFF', 'Value'], ",.4f")
        self.update_table_cell("Terminal Value", terminal_col, valuation.loc['Terminal Value', 'Value'], ",.4f")
        self.update_table_cell("Discounted Terminal Value", terminal_col, 
                             valuation.loc['Discounted Terminal Value', 'Value'], ",.4f")
        
        # Update summary values (column 0)
        self.update_table_cell("Sum of Discounted FCFF", 0, valuation.loc['Sum of Discounted FCFFs', 'Value'], ",.4f")
        self.update_table_cell("Total Firm Value", 0, valuation.loc['Total Firm Value', 'Value'], ",.4f")
        self.update_table_cell("Equity Value", 0, valuation.loc['Equity Value', 'Value'], ",.4f")
        self.update_table_cell("Fair Value per Share", 0, valuation.loc['Fair Value per Share', 'Value'], ",.4f")
        
        # Update terminal ROIC
        self.update_table_cell("Return on Invested Capital", terminal_col, user_input['roic_tv'], ".4f")
    
    def update_table_cell(self, row_name, col, value, format_str):
        """Helper method to safely update table cells"""
        row = self.find_row(row_name)
        if row >= 0:
            try:
                if isinstance(value, (int, float)):
                    self.table.item(row, col).setText(format(value, format_str).format(value))
            except Exception as e:
                print(f"Error updating cell {row_name}[{col}]: {str(e)}")
    
    def parse_percentage_row(self, row_name, include_terminal=False):
        row = self.find_row(row_name)
        if row == -1:
            raise ValueError(f"Row '{row_name}' not found")
        
        values = []
        end_col = 12 if include_terminal else 11
        
        for col in range(1, end_col):
            item = self.table.item(row, col)
            if not item or not item.text():
                raise ValueError(f"Missing value in {row_name}, year {col}")
            
            try:
                text = item.text().replace('%', '')
                value = float(text)
                if abs(value) > 1 and not text.endswith('%'):
                    value /= 100  # Convert to decimal if >1 without % sign
                values.append(value)
            except ValueError:
                raise ValueError(f"Invalid decimal value in {row_name}, year {col}")
        
        return values

    def parse_float_row(self, row_name, include_terminal=False):
        row = self.find_row(row_name)
        if row == -1:
            raise ValueError(f"Row '{row_name}' not found")
        
        values = []
        end_col = 12 if include_terminal else 11
        
        for col in range(1, end_col):
            item = self.table.item(row, col)
            if not item or not item.text():
                if row_name == "Reinvestment Rate" and col == 11:
                    continue  # Skip terminal year for reinvestment rate
                raise ValueError(f"Missing value in {row_name}, year {col}")
            
            try:
                values.append(float(item.text()))
            except ValueError:
                raise ValueError(f"Invalid numeric value in {row_name}, year {col}")
        
        return values

    def find_row(self, row_name):
        for row in range(self.table.rowCount()):
            if self.table.verticalHeaderItem(row).text().strip() == row_name.strip():
                return row
        return -1    

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding plots"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

class HistoricalDataTab(QWidget):
    """Modernized Historical Data Tab with proper plot updates"""
    def __init__(self, parent=None):
        super(HistoricalDataTab, self).__init__(parent)
        self.parent = parent
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(12)
        
        # Modern header with company name
        self.header = QLabel("Historical Financial Overview")
        self.header.setFont(QFont('Segoe UI', 14, QFont.Bold))
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(f"""
            color: {COLORS['primary']};
            padding: 12px;
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 8px;
        """)
        self.layout.addWidget(self.header)
        
        # Create scroll area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f8f9fa;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #d1d5db;
                min-height: 20px;
                border-radius: 4px;
            }
        """)
        
        # Content widget for scroll area
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(16)
        self.grid_layout.setContentsMargins(8, 8, 8, 8)
        
        scroll.setWidget(self.scroll_content)
        self.layout.addWidget(scroll)
        
        # Create directory for saving plots
        self.plots_dir = "saved_plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Store current company and data
        self.current_company = None
        self.current_fcff = None
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Charts")
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                min-height: 36px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary']};
            }}
        """)
        self.refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.refresh_btn.clicked.connect(self._force_refresh)
        
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.addStretch()
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addStretch()
        self.layout.addWidget(btn_container)
    
    def update_plots(self, fcff):
        """Update the plots with data for the selected stock"""
        try:
            self.current_fcff = fcff
            company = "Microsoft" if fcff.stock == "MSFT" else "NVIDIA"
            
            # Update header
            self.header.setText(f"Historical Financial Overview - {company}")
            
            # Only regenerate plots if company changed or data is stale
            if company != self.current_company or not self._plots_exist(company):
                self.current_company = company
                self._generate_and_display_plots(fcff, company)
            
        except Exception as e:
            print(f"Error updating plots: {e}")
            self._show_error_message(str(e))
    
    def _force_refresh(self):
        """Force refresh all plots"""
        if self.current_fcff and self.current_company:
            self._generate_and_display_plots(self.current_fcff, self.current_company)
    
    def _plots_exist(self, company):
        """Check if plot files exist for current company"""
        required_files = [
            f"{company}_reinvestment.png",
            f"{company}_revenue.png",
            f"{company}_ebit.png",
            f"{company}_invested_capital.png",
            f"{company}_operating_margin.png",
            f"{company}_stock_price.png"
        ]
        return all(os.path.exists(os.path.join(self.plots_dir, f)) for f in required_files)
    
    def _generate_and_display_plots(self, fcff, company):
        """Generate and display all plots"""
        try:
            # Clear previous plots
            self._clear_grid_layout()
            
            # Generate and save plots
            plot_files = self._generate_and_save_plots(fcff, company)
            
            # Arrange in grid with proper sizing
            self._arrange_plots_in_grid(plot_files)
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            self._show_error_message(str(e))
    
    def _clear_grid_layout(self):
        """Clear all widgets from the grid layout"""
        for i in reversed(range(self.grid_layout.count())): 
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
    
    def _generate_and_save_plots(self, fcff, company):
        """Generate all plots and save them to files"""
        plot_files = {}
        
        try:
            # Generate and save each plot with consistent sizing
            fig, ax = fcff.plot_reinvestment_only()
            fig.set_size_inches(6, 4)  # Standard size for all plots
            reinvestment_path = self._save_plot(fig, f"{company}_reinvestment.png")
            plt.close(fig)
            
            fig, ax = fcff.plot_revenue_and_growth()
            fig.set_size_inches(6, 4)
            revenue_path = self._save_plot(fig, f"{company}_revenue.png")
            plt.close(fig)
            
            fig, ax = fcff.plot_ebit()
            fig.set_size_inches(6, 4)
            ebit_path = self._save_plot(fig, f"{company}_ebit.png")
            plt.close(fig)
            
            fig, ax1, ax2 = fcff.plot_invested_capital_and_roic()
            fig.set_size_inches(6, 4)  # Same size even for dual-axis plots
            invested_capital_path = self._save_plot(fig, f"{company}_invested_capital.png")
            plt.close(fig)
            
            fig, ax = fcff.plot_operating_margin()
            fig.set_size_inches(6, 4)
            operating_margin_path = self._save_plot(fig, f"{company}_operating_margin.png")
            plt.close(fig)
            
            fig, ax = fcff.plot_stock_price()
            fig.set_size_inches(6, 4)
            stock_price_path = self._save_plot(fig, f"{company}_stock_price.png")
            plt.close(fig)
            
            return {
                'reinvestment': reinvestment_path,
                'revenue': revenue_path,
                'ebit': ebit_path,
                'invested_capital': invested_capital_path,
                'operating_margin': operating_margin_path,
                'stock_price': stock_price_path
            }
            
        except Exception as e:
            # Clean up any partial files if error occurs
            for path in plot_files.values():
                if os.path.exists(path):
                    os.remove(path)
            raise
    
    def _save_plot(self, fig, filename):
        """Save a matplotlib figure to file with modern styling"""
        filepath = os.path.join(self.plots_dir, filename)
        
        # Apply modern styling before saving
        fig.patch.set_facecolor('#f8f9fa')
        for ax in fig.get_axes():
            ax.set_facecolor('#ffffff')
            ax.grid(color='#e9ecef', linestyle='-', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#adb5bd')
            ax.spines['left'].set_color('#adb5bd')
        
        fig.savefig(filepath, bbox_inches='tight', dpi=120, facecolor=fig.get_facecolor())
        return filepath
    
    def _arrange_plots_in_grid(self, plot_files):
        """Arrange the saved plots in a balanced grid layout"""
        try:
            # Create plot cards with consistent sizing
            self._create_plot_card(plot_files['reinvestment'], 0, 0, "Reinvestment")
            self._create_plot_card(plot_files['revenue'], 0, 1, "Revenue and Growth")
            self._create_plot_card(plot_files['ebit'], 0, 2, "EBIT")
            
            # Make the invested capital plot span two columns
            self._create_plot_card(plot_files['invested_capital'], 1, 0, "Invested Capital & ROIC")
            self._create_plot_card(plot_files['stock_price'], 1, 1, "Stock Price")
            
            # Add operating margin below if needed
            self._create_plot_card(plot_files['operating_margin'], 1, 2, "Operating Margin")
            
        except Exception as e:
            print(f"Error arranging plots: {e}")
            self._show_error_message(str(e))
    
    def _create_plot_card(self, filepath, row, col, title, rowspan=1, colspan=1):
        """Create a card container for a plot with proper sizing"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Plot file not found: {filepath}")
            
            # Create card container
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: white;
                    border-radius: 8px;
                    border: 1px solid #e0e0e0;
                }}
            """)
            card.setMinimumSize(400, 300)
            
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)
            
            # Title label
            title_label = QLabel(title)
            title_label.setFont(QFont('Segoe UI', 10, QFont.Bold))
            title_label.setStyleSheet(f"color: {COLORS['primary']};")
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)
            
            # Plot container
            plot_container = QWidget()
            plot_container.setStyleSheet("background-color: white;")
            plot_layout = QVBoxLayout(plot_container)
            plot_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create image label for the plot
            plot_label = QLabel()
            plot_label.setAlignment(Qt.AlignCenter)
            plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # Load and display the saved plot
            pixmap = QPixmap(filepath)
            if not pixmap.isNull():
                # Scale pixmap to fit while maintaining aspect ratio
                plot_label.setPixmap(pixmap.scaled(
                    plot_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
            else:
                raise ValueError("Could not load plot image")
            
            plot_layout.addWidget(plot_label)
            layout.addWidget(plot_container, 1)
            
            # Add to grid with specified span
            self.grid_layout.addWidget(card, row, col, rowspan, colspan)
            
            # Ensure the plot label updates on resize
            plot_label.resizeEvent = lambda event: self._update_plot_size(plot_label, filepath)
            
        except Exception as e:
            print(f"Error creating plot card: {e}")
            error_label = QLabel(f"Could not load {title} chart")
            error_label.setStyleSheet(f"color: {COLORS['danger']};")
            error_label.setAlignment(Qt.AlignCenter)
            if 'layout' in locals():
                layout.addWidget(error_label)
    
    def _update_plot_size(self, label, filepath):
        """Update plot size when container is resized"""
        if os.path.exists(filepath):
            pixmap = QPixmap(filepath)
            if not pixmap.isNull():
                label.setPixmap(pixmap.scaled(
                    label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                ))
    
    def _show_error_message(self, message):
        """Show an error message in the grid"""
        self._clear_grid_layout()
        
        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        
        error_label = QLabel("Error loading historical data")
        error_label.setFont(QFont('Segoe UI', 12, QFont.Bold))
        error_label.setStyleSheet(f"color: {COLORS['danger']};")
        error_label.setAlignment(Qt.AlignCenter)
        error_layout.addWidget(error_label)
        
        detail_label = QLabel(message)
        detail_label.setWordWrap(True)
        detail_label.setAlignment(Qt.AlignCenter)
        error_layout.addWidget(detail_label)
        
        self.grid_layout.addWidget(error_widget, 0, 0, 1, 3)

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