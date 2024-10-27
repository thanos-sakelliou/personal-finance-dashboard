import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import argparse


def get_color(category):
    # Define an improved color map for categories and subcategories
    color_map = {
        "Savings": "#0bb4ff",
        "Income": "#50e991",
        "Transportation": "#9b19f5",
        "Uncategorised": "#d7e1ee",
        "Housing": "#ffa300",
        "Electricity": "#e6d800",
        "Shopping": "#dc0ab4",
        "Food & Consumables": "#e60049",
        "Subscriptions": "#f46a9b",
        "Health": "#7c1158",
        "Taxes & obligations": "#b3d4ff",
        "Entertainment": "#fdcce5",
        "Travels": "#00bfa0",
    }
    return color_map.get(category, "#d3d3d3")  # Default to light grey if not in map


def create_balance_chart(df, threshold=100):
    # Create the base line chart for the account balance
    line_trace = go.Scatter(
        x=df["Transaction Date"],
        y=df["Progressive Balance"],
        mode="lines",
        name="Balance",
        line=dict(color="blue"),
    )

    # Filter high-value transactions
    high_value_df = df[abs(df["Amount"]) >= threshold]

    # Iterate through the categories and create scatter plots for high-value transactions
    scatter_traces = []
    for category in high_value_df["Supercategory"].unique():
        category_df = high_value_df[high_value_df["Supercategory"] == category]
        color = get_color(category)

        scatter_trace = go.Scatter(
            x=category_df["Transaction Date"],
            y=category_df["Progressive Balance"],
            mode="markers",
            name=category,
            marker=dict(color=color, size=10, symbol="circle"),
            text=[
                f"{desc}<br>" f"{date}<br>" f"<b>{amount}€<br></b>" f"{comments.split('_')[:1]}"
                for desc, date, amount, comments in zip(
                    category_df["Transaction Description"],
                    category_df["Transaction Date"].dt.strftime("%d/%m/%Y"),
                    category_df["Amount"],
                    category_df["Comments"],
                )
            ],
            hoverinfo="text",
        )
        scatter_traces.append(scatter_trace)

    # Combine the line and scatter plots
    fig = go.Figure(data=[line_trace] + scatter_traces)

    # Update layout
    fig.update_layout(
        title="Blow Account Balance",
        xaxis_title="Date",
        yaxis_title="Account Balance (€)",
        showlegend=True,
    )

    return fig


def create_monthly_balance_change_bar_chart(df):
    # Extract Year-Month for grouping by month
    df["Year-Month"] = df["Transaction Date"].dt.to_period("M")

    # Calculate the account balance at the end of each month
    # Identify the last transaction for each month without altering the order of the DataFrame
    monthly_indices = df.groupby("Year-Month").head(1).index

    # Extract these rows while preserving the order of appearance in the original DataFrame
    monthly_balance = df.loc[monthly_indices, ["Year-Month", "Progressive Balance"]].reset_index(drop=True)

    # Calculate monthly changes in balance
    monthly_balance["Monthly Change"] = monthly_balance["Progressive Balance"][::-1].diff()[::-1].fillna(0)

    # Round the values for clarity
    monthly_balance["Progressive Balance"] = monthly_balance["Progressive Balance"].round(0)
    monthly_balance["Monthly Change"] = monthly_balance["Monthly Change"].round(0)

    # Define colors for the bar chart (green for positive, red for negative)
    monthly_balance["Color"] = monthly_balance["Monthly Change"].apply(lambda x: "green" if x >= 0 else "red")

    # Bar chart for monthly changes in balance
    bar_trace = go.Bar(
        x=monthly_balance["Year-Month"].astype(str),
        y=monthly_balance["Monthly Change"],
        marker_color=monthly_balance["Color"],
        text=monthly_balance["Monthly Change"],
        textposition="auto",
        name="Monthly Balance Change",
    )

    # Line chart for the account balance at the end of each month
    line_trace = go.Scatter(
        x=monthly_balance["Year-Month"].astype(str),
        y=monthly_balance["Progressive Balance"],
        mode="lines+markers",
        name="Account Balance (Month-End)",
        line=dict(color="blue", width=2),
        marker=dict(size=8, symbol="circle"),
    )

    # Combine the bar and line chart
    fig = go.Figure(data=[bar_trace, line_trace])

    # Update layout
    fig.update_layout(
        title="Monthly Gains and losses",
        xaxis_title="Month",
        yaxis_title="Amount (€)",
        showlegend=True,
    )

    return fig


def create_savings_goal_gauge(df, savings_goal=10000):
    # Calculate remaining money at the end of the period (final balance)
    total_balance = -df[df["Supercategory"] == "Savings"]["Amount"].sum()

    # Create a gauge chart to show progress towards savings goal
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=total_balance,
            gauge={"axis": {"range": [0, savings_goal]}, "bar": {"color": "green"}},
            title={"text": "Savings Goal Progress"},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    return fig


def create_monthly_expenses_bar_chart(df):
    expenses_df = df[df["Amount"] < 0].copy()
    expenses_df["Month"] = expenses_df["Transaction Date"].dt.to_period("M").astype(str)

    # Negate the amount values to make the bars point upwards
    expenses_df["Amount"] = -expenses_df["Amount"]

    monthly_expenses = expenses_df.groupby(["Month", "Supercategory"])["Amount"].sum().reset_index()

    fig = px.bar(
        monthly_expenses,
        x="Month",
        y="Amount",
        color="Supercategory",
        title="Monthly Expenses by Category",
        labels={"Amount": "Total Expenses (€)", "Month": "Month"},
        color_discrete_map={sc: get_color(sc) for sc in monthly_expenses["Supercategory"].unique()},
    )
    fig.update_layout(barmode="stack", xaxis_title="Month", yaxis_title="Total Expenses (€)")
    return fig


def create_all_time_expenses_bar_chart(df):
    expenses_df = df[df["Amount"] < 0].copy()

    # Negate the amount values to make the bars point upwards
    expenses_df["Amount"] = -expenses_df["Amount"]

    total_expenses = expenses_df.groupby("Supercategory")["Amount"].sum().reset_index()

    fig = px.bar(
        total_expenses,
        x="Supercategory",
        y="Amount",
        color="Supercategory",
        title="All-Time Expenses by Category",
        labels={"Amount": "Total Expenses (€)", "Supercategory": "Supercategory"},
        color_discrete_map={sc: get_color(sc) for sc in total_expenses["Supercategory"].unique()},
    )
    fig.update_layout(barmode="stack", xaxis_title="Supercategory", yaxis_title="Total Expenses (€)")
    return fig


class PersonalFinanceDashboard:
    def __init__(self, data_file, add_file):
        self.app = dash.Dash(__name__)
        self.data_file = data_file
        self.loader = Piraeus_data_loader(self.data_file)
        if add_file:
            self.loader.digest_bank_statement(add_file)
        self.expenses_df = self.loader.get_expenses_df()
        self.balance_df = self.loader.get_balance_df()

        # Define the layout in the constructor
        self.app.layout = html.Div(
            [
                html.H1("Personal Finance Dashboard"),
                dcc.Graph(id="balance-chart"),
                dcc.Graph(id="monthly-balance-change-bar-chart"),
                dcc.Graph(id="monthly-expenses-bar-chart"),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="savings-goal-gauge")],
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        html.Div(
                            [dcc.Graph(id="all-time-expenses-bar-chart")],
                            style={"width": "68%", "display": "inline-block"},
                        ),
                    ],
                    style={"display": "flex", "justify-content": "space-between"},
                ),
                html.H3("Uncategorized Transactions"),
                html.Div(
                    id="uncategorized-list",
                    style={"max-height": "400px", "overflow-y": "scroll"},
                ),
                html.H3("Batch Update Supercategory by Date Range"),
                html.Div(
                    [
                        dcc.DatePickerRange(
                            id="date-range-picker",
                            display_format="DD/MM/YYYY",
                        ),
                        dcc.Dropdown(
                            id="supercategory-dropdown",
                            options=[
                                {"label": cat, "value": cat}
                                for cat in [
                                    "Savings",
                                    "Income",
                                    "Transportation",
                                    "Housing",
                                    "Electricity",
                                    "Shopping",
                                    "Food & Consumables",
                                    "Subscriptions",
                                    "Health",
                                    "Taxes & obligations",
                                    "Entertainment",
                                    "Travels",
                                ]
                            ],
                            placeholder="Select Supercategory",
                            style={"width": "200px"},
                        ),
                        html.Button("Set Supercategory", id="set-supercategory-button"),
                    ],
                    style={"display": "flex", "align-items": "center"},
                ),
                html.Button("Save Changes", id="save-button", n_clicks=0),
                html.Button("Reload Data", id="reload-button", n_clicks=0),
            ]
        )

        # Define the callbacks
        self.define_callbacks()

    def define_callbacks(self):
        @self.app.callback(
            Output("uncategorized-list", "children"),
            [
                Input("reload-button", "n_clicks"),
                Input(
                    {
                        "type": "set-supercategory",
                        "index": dash.dependencies.ALL,
                        "category": dash.dependencies.ALL,
                    },
                    "n_clicks",
                ),
            ],
            State("uncategorized-list", "children"),
        )
        def load_or_update_uncategorized_list(reload_n_clicks, n_clicks_list, children):
            ctx = dash.callback_context  # Get the context of the callback to know which input triggered it

            if not ctx.triggered:
                # If no input has triggered, return the current children
                return children

            triggered_input = ctx.triggered[0]["prop_id"]

            # If Reload button was clicked
            if "reload-button" in triggered_input:
                self.loader = Piraeus_data_loader(self.data_file)
                self.expenses_df = self.loader.get_expenses_df()
                self.balance_df = self.loader.get_balance_df()

                uncategorized_df = self.loader.df[
                    (self.loader.df["Supercategory"] == "Uncategorised") | (self.loader.df["Supercategory"] == "Other")
                ]
                uncategorized_df = uncategorized_df[abs(uncategorized_df["Amount"]) > 5]
                items = self.create_uncategorized_items(uncategorized_df)
                return items

            # If a category button was clicked
            elif "set-supercategory" in triggered_input:
                for button_id, n_clicks in enumerate(n_clicks_list):
                    if n_clicks:
                        button_data = ctx.triggered[0]["prop_id"].split(".")[0]
                        button_info = eval(button_data)  # This is a dict with `type`, `index`, and `category`
                        index = button_info["index"]
                        new_category = button_info["category"]

                        # Update the supercategory for the corresponding row
                        self.loader.df.at[index, "Supercategory"] = new_category
                        print(self.loader.df.iloc[index])

                # Reload the uncategorized transactions and rerender the list
                uncategorized_df = self.loader.df[
                    (self.loader.df["Supercategory"] == "Uncategorised") | (self.loader.df["Supercategory"] == "Other")
                ]
                uncategorized_df = uncategorized_df[abs(uncategorized_df["Amount"]) > 5]
                items = self.create_uncategorized_items(uncategorized_df)
                return items

            # Return the existing children if none of the above is triggered
            return children

        @self.app.callback(
            Input("set-supercategory-button", "n_clicks"),
            [
                State("date-range-picker", "start_date"),
                State("date-range-picker", "end_date"),
                State("supercategory-dropdown", "value"),
            ],
        )
        def batch_update_supercategory(n_clicks, start_date, end_date, supercategory):
            if n_clicks and start_date and end_date and supercategory:
                # Update transactions within the date range
                mask = (
                    (self.loader.df["Transaction Date"] >= pd.to_datetime(start_date))
                    & (self.loader.df["Transaction Date"] <= pd.to_datetime(end_date))
                    & (
                        self.loader.df[
                            self.loader.df["Supercategory"].isin(
                                [
                                    "Uncategorised",
                                    "Shopping",
                                    "Transportation",
                                    "Food & Consumables",
                                    "Health",
                                    "Entertainment",
                                    "Other",
                                ]
                            )
                        ]
                    )
                )
                self.loader.df.loc[mask, "Supercategory"] = supercategory

                uncategorized_df = self.loader.df[self.loader.df["Supercategory"] == "Uncategorised"]
                uncategorized_df = uncategorized_df[abs(uncategorized_df["Amount"]) > 5]
                items = self.create_uncategorized_items(uncategorized_df)
                return items

            return dash.no_update

        @self.app.callback(
            [
                Output("balance-chart", "figure"),
                Output("monthly-balance-change-bar-chart", "figure"),
                Output("savings-goal-gauge", "figure"),
                Output("monthly-expenses-bar-chart", "figure"),
                Output("all-time-expenses-bar-chart", "figure"),
            ],
            [Input("balance-chart", "id")],
        )
        def update_graphs(_):
            balance_chart = create_balance_chart(self.balance_df, threshold=50)
            monthly_balance_change_bar_chart = create_monthly_balance_change_bar_chart(self.expenses_df)
            savings_goal_gauge = create_savings_goal_gauge(self.balance_df)
            monthly_expenses_bar_chart = create_monthly_expenses_bar_chart(self.expenses_df)
            all_time_expenses_bar_chart = create_all_time_expenses_bar_chart(self.expenses_df)
            return (
                balance_chart,
                monthly_balance_change_bar_chart,
                savings_goal_gauge,
                monthly_expenses_bar_chart,
                all_time_expenses_bar_chart,
            )

        @self.app.callback(Output("save-button", "n_clicks"), Input("save-button", "n_clicks"))
        def save_data(n_clicks):
            if n_clicks > 0:
                self.loader.save_to_excel()
            return n_clicks

    def create_uncategorized_items(self, df):
        items = []
        for i, row in df.iterrows():
            items.append(
                html.Div(
                    [
                        html.Span(
                            row["Transaction Date"].strftime("%d/%m/%Y"),
                            style={
                                "flex": "1",
                                "max-width": "100px",
                                "overflow": "visible",
                                "white-space": "normal",
                                "word-wrap": "break-word",
                            },
                        ),
                        html.Span(
                            row["Transaction Description"],
                            style={"flex": "1", "max-width": "250px"},
                        ),
                        html.Span(
                            f'{row["Amount"]}€',
                            style={"flex": "1", "max-width": "100px"},
                        ),
                        html.Span(
                            f'{row["Comments"]}',
                            style={"flex": "1", "max-width": "400px"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Travels",
                                    id={
                                        "type": "set-supercategory",
                                        "index": i,
                                        "category": "Travels",
                                    },
                                    className="button",
                                    style={
                                        "backgroundColor": get_color("Travels"),
                                        "marginLeft": "5px",
                                    },
                                ),
                                html.Button(
                                    "Food & Consumables",
                                    id={
                                        "type": "set-supercategory",
                                        "index": i,
                                        "category": "Food & Consumables",
                                    },
                                    className="button",
                                    style={
                                        "backgroundColor": get_color("Food & Consumables"),
                                        "marginLeft": "5px",
                                    },
                                ),
                                html.Button(
                                    "Transportation",
                                    id={
                                        "type": "set-supercategory",
                                        "index": i,
                                        "category": "Transportation",
                                    },
                                    className="button",
                                    style={
                                        "backgroundColor": get_color("Transportation"),
                                        "marginLeft": "5px",
                                    },
                                ),
                                html.Button(
                                    "Health",
                                    id={
                                        "type": "set-supercategory",
                                        "index": i,
                                        "category": "Health",
                                    },
                                    className="button",
                                    style={
                                        "backgroundColor": get_color("Health"),
                                        "marginLeft": "5px",
                                    },
                                ),
                                html.Button(
                                    "Savings",
                                    id={
                                        "type": "set-supercategory",
                                        "index": i,
                                        "category": "Savings",
                                    },
                                    className="button",
                                    style={
                                        "backgroundColor": get_color("Savings"),
                                        "marginLeft": "5px",
                                    },
                                ),
                                html.Button(
                                    "Taxes & obligations",
                                    id={
                                        "type": "set-supercategory",
                                        "index": i,
                                        "category": "Taxes & obligations",
                                    },
                                    className="button",
                                    style={
                                        "backgroundColor": get_color("Taxes & obligations"),
                                        "marginLeft": "5px",
                                    },
                                ),
                            ],
                            style={"display": "flex", "align-items": "center"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "align-items": "flex-start",
                        "padding": "5px 0",
                        "font-family": "'Roboto', sans-serif",
                    },
                )
            )
        return items

    def run(self):
        self.app.run_server(debug=True)


class Piraeus_data_loader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self.load_from_excel()

    def load_from_excel(self):
        # Load the clean data from the Excel file
        df = pd.read_excel(self.filepath, header=0)
        self.column_names = df.columns.tolist()
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], format="%d/%m/%Y")  # .dt.strftime('%d/%m/%Y')
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        df["Progressive Balance"] = pd.to_numeric(df["Progressive Balance"], errors="coerce")
        return df

    def digest_bank_statement(self, filepath):
        # Load additional transactions from another Excel file
        new_df = pd.read_excel(filepath, skiprows=5, skipfooter=3)

        # Check if 'Progressive Balance' is present in the new data
        if len(new_df.columns) == 6:
            new_df.columns = self.column_names[:-2]
            new_df["Progressive Balance"] = None  # Placeholder for now
        else:
            if len(new_df.columns) == 8:
                new_df = new_df.drop(new_df.columns[-1], axis=1)
            new_df.columns = self.column_names[:-1]
            new_df["Progressive Balance"] = pd.to_numeric(new_df["Progressive Balance"], errors="coerce")

        # Convert types
        new_df["Transaction Date"] = pd.to_datetime(
            new_df["Transaction Date"], format="%d/%m/%Y"
        )  # .dt.strftime('%d/%m/%Y')
        new_df["Amount"] = pd.to_numeric(new_df["Amount"], errors="coerce")

        # add supercategory column
        new_df = self.map_supercategories(new_df)

        # Determine where the new data should be appended
        if new_df["Transaction Date"].max() <= self.df["Transaction Date"].max():
            combined_df = (
                pd.concat([self.df, new_df])
                .drop_duplicates(subset=["Transaction Date", "Amount", "Comments"])
                .reset_index(drop=True)
            )
        elif new_df["Transaction Date"].min() >= self.df["Transaction Date"].min():
            # If all dates in df2 are after the last date in df1, append df2 at the bottom
            combined_df = (
                pd.concat([new_df, self.df])
                .drop_duplicates(subset=["Transaction Date", "Amount", "Comments"])
                .reset_index(drop=True)
            )
        else:
            # If there's overlap in the date ranges, raise an error or handle accordingly
            raise ValueError("The dates in df2 overlap with the dates in df1")

        # Calculate the progressive balance if missing
        if combined_df["Progressive Balance"].isna().any():
            combined_df = self.calculate_progressive_balance(combined_df)

        self.df = combined_df

    def save_to_excel(self):
        export_df = self.df
        export_df["Transaction Date"] = export_df["Transaction Date"].dt.strftime("%d/%m/%Y")
        export_df.to_excel(self.filepath, index=False)

    def calculate_progressive_balance(self, df):
        if "Progressive Balance" not in df.columns:
            df["Progressive Balance"] = np.nan

        # Fill the missing 'Progressive Balance' values based on available data
        for i in range(len(df)):
            if pd.isna(df.loc[i, "Progressive Balance"]):
                prev_balance = df.loc[i - 1, "Progressive Balance"] if i > 0 else np.nan
                # next_balance = df.loc[i + 1, 'Progressive Balance'] if i < len(df) - 1 else np.nan
                prev_amount = df.loc[i - 1, "Amount"]

                df.loc[i, "Progressive Balance"] = prev_balance - prev_amount
        return df

    def update_progressive_balance(self, df):
        # Start with the last transaction and work backwards
        for i in range(len(df) - 2, -1, -1):
            prev_balance = df.iloc[i + 1]["Progressive Balance"]
            amount = df.iloc[i]["Amount"]
            df.iloc[i, df.columns.get_loc("Progressive Balance")] = prev_balance + amount

        return df

    def get_balance_df(self):
        return self.df

    def get_expenses_df(self):
        # Exclude the "Savings" category
        expenses_df = self.df[self.df["Supercategory"] != "Savings"]
        expenses_df = expenses_df.reset_index(drop=True)
        expenses_df = self.update_progressive_balance(expenses_df)
        return expenses_df

    def map_supercategories(self, df):
        # Define the category to supercategory mapping
        category_mapping = {
            "Ανακατανομή": "Savings",
            "Έσοδα / Έσοδα": "Income",
            "Μετακίνηση / Αυτοκίνητο": "Transportation",
            "Μετακίνηση / Δρομολόγια": "Transportation",
            "Μετακίνηση / Εισιτήρια": "Transportation",
            "Μετακίνηση / Parking": "Transportation",
            "Μετακίνηση / Καύσιμα": "Transportation",
            "Μετρητά / Αναλήψεις": "Uncategorised",
            "Μετρητά / Εμβάσματα": "Uncategorised",
            "Μετρητά / Μεταφορές": "Uncategorised",
            "Σπίτι / Άλλο": "Uncategorised",
            "Σπίτι / Ενοίκιο": "Housing",
            "Σπίτι / Ενέργεια": "Electricity",
            "Σπίτι / Εξοπλισμός": "Shopping",
            "Σπίτι / Supermarket": "Food & Consumables",
            "Σπίτι / Επικοινωνίες": "Subscriptions",
            "Υγεία / Ιατρικά": "Health",
            "Υγεία / Υπηρεσίες": "Health",
            "Υγεία / Φαρμακεία": "Health",
            "Υποχρεώσεις / Προμήθειες": "Uncategorised",
            "Υποχρεώσεις / Φόροι": "Taxes & obligations",
            "Υποχρεώσεις / Υπηρεσίες": "Taxes & obligations",
            "Ψυχαγωγία / Άθληση": "Subscriptions",
            "Ψυχαγωγία / Διασκέδαση": "Entertainment",
            "Ψυχαγωγία / Εστιατόρια": "Food & Consumables",
            "Ψυχαγωγία / Καθημερινά": "Food & Consumables",
            "Ψυχαγωγία / Συνδρομές": "Subscriptions",
            "Ψυχαγωγία / Ταξίδια": "Travels",
            "Ψυχαγωγία / Χόμπυ": "Shopping",
            "Ψώνια / Αξεσουάρ": "Shopping",
            "Ψώνια / Άλλο": "Shopping",
            "Ψώνια / Ηλεκτρονικά": "Shopping",
            "Ψώνια / Ρουχισμός": "Shopping",
            "Χωρίς Κατηγορία": "Uncategorised",
        }

        # Define keywords for specific supercategories
        keyword_mapping = {
            "Subscriptions": ["spotify", "audible", "netflix"],
            "Savings": ["revolut"],
        }

        # Function to check if any keyword matches in the description
        def check_keywords(description):
            description = description.lower()  # Convert to lowercase for case-insensitive matching
            for supercategory, keywords in keyword_mapping.items():
                if any(keyword in description for keyword in keywords):
                    return supercategory
            return None

        # Define a function to apply the mapping and rules based on transaction description and amount
        def apply_rules(row):
            supercategory_from_keywords = check_keywords(row["Comments"])
            if supercategory_from_keywords:
                return supercategory_from_keywords
            elif (abs(row["Amount"]) < 40) and (
                row["Transaction Description"].strip() in ["ΕΞΕΡΧΟΜΕΝΟ ΕΜΒΑΣΜΑ", "ΕΙΣΕΡΧΟΜΕΝΟ ΕΜΒΑΣΜΑ"]
            ):
                return "Food & Consumables"
            return category_mapping.get(row["Category"], "Other")

        df["Supercategory"] = df.apply(apply_rules, axis=1)

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Personal Finance Dashboard.")
    parser.add_argument(
        "--file",
        type=str,
        default="PiraeusStatement.xlsx",
        help="Path to the Excel file containing transaction data",
    )
    parser.add_argument("--add", type=str, help="Extra transactions to digest")
    args = parser.parse_args()

    dashboard = PersonalFinanceDashboard(args.file, args.add)
    dashboard.run()

# TODO bigger graph.
# TODO remove or change or add the trend heatmap
# TODO add more history. complete the balance from xls that doesnt have
# TODO nice graph for the categories. fix the food cat with small incomes

# TODO uncategorised easy categorization
# TODO filter some automatically


# TODO when editing data it should be updated. neither "update" works
# TODO update travels from google maps timeline
