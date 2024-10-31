from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Paths for the dataset and folders for plots
DATA_PATH = "C://Users//bella//OneDrive//Dokumen//Supermart Grocery Sales - Retail Analytics Dataset.csv"
app.config['PLOT_FOLDER'] = 'static/plots/'
os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)

# Load the dataset
df = pd.read_csv(DATA_PATH)
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df.dropna(subset=['Order Date'], inplace=True)
df['Month'] = df['Order Date'].dt.strftime('%B')
df['Year'] = df['Order Date'].dt.year

# Helper functions to generate plots
def create_bar_plot(data, title):
    if data.empty:
        raise ValueError("No data available for the selected categories.")

    plot_file = os.path.join(app.config['PLOT_FOLDER'], 'bar_plot.png')
    if os.path.exists(plot_file):
        os.remove(plot_file)

    plt.figure(figsize=(12, 6))
    sales_data = data.groupby('Category')['Sales'].sum().reset_index()
    sns.barplot(data=sales_data, x='Category', y='Sales', palette='viridis', edgecolor='black')
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    return plot_file

def create_pie_chart(data, title):
    if data.empty:
        raise ValueError("No data available for the selected categories.")

    plot_file = os.path.join(app.config['PLOT_FOLDER'], 'pie_chart.png')
    if os.path.exists(plot_file):
        os.remove(plot_file)

    plt.figure(figsize=(8, 8))
    sales_data = data.groupby('Category')['Sales'].sum()
    plt.pie(sales_data, labels=sales_data.index, autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    return plot_file

# Route for the main page
@app.route('/', methods=['GET'])
def index():
    months = df['Month'].unique()
    years = df['Year'].unique()
    states = df['State'].unique()
    cities = df['City'].unique()
    return render_template('index.html', months=months, years=years, states=states, cities=cities)

# Route for month view
@app.route('/month_view', methods=['POST'])
def month_view():
    selected_month = request.form.get('month')
    selected_year = int(request.form.get('year'))
    filtered_data = df[(df['Month'] == selected_month) & (df['Year'] == selected_year)]
    chart_type = request.form.get('chart_type')

    title = f'Sales by Category for {selected_month} {selected_year}'
    plot_file = create_bar_plot(filtered_data, title) if chart_type == 'bar' else create_pie_chart(filtered_data, title)

    return render_template('result.html', plot_file=plot_file)

# Route for year view
@app.route('/year_view', methods=['POST'])
def year_view():
    selected_year = int(request.form.get('year'))
    filtered_data = df[df['Year'] == selected_year]
    chart_type = request.form.get('chart_type')

    title = f'Sales by Category for the Year {selected_year}'
    plot_file = create_bar_plot(filtered_data, title) if chart_type == 'bar' else create_pie_chart(filtered_data, title)

    return render_template('result.html', plot_file=plot_file)

# Route for state view
@app.route('/state_view', methods=['POST'])
def state_view():
    selected_state = request.form.get('state')
    filtered_data = df[df['State'] == selected_state]
    chart_type = request.form.get('chart_type')

    title = f'Sales by Category for State {selected_state}'
    plot_file = create_bar_plot(filtered_data, title) if chart_type == 'bar' else create_pie_chart(filtered_data, title)

    return render_template('result.html', plot_file=plot_file)

# Route for city view
@app.route('/city_view', methods=['POST'])
def city_view():
    selected_city = request.form.get('city')
    filtered_data = df[df['City'] == selected_city]
    chart_type = request.form.get('chart_type')

    title = f'Sales by Category for City {selected_city}'
    plot_file = create_bar_plot(filtered_data, title) if chart_type == 'bar' else create_pie_chart(filtered_data, title)

    return render_template('result.html', plot_file=plot_file)

if __name__ == '__main__':
    app.run(debug=True)
