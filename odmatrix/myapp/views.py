import pandas as pd
import matplotlib
from django.shortcuts import render
from statsmodels.tsa.arima.model import ARIMA  # Uncommented the ARIMA import
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from django.shortcuts import render, HttpResponse
import io, os
import urllib, base64
from django.conf import settings
from .pyspark_config import get_spark_session
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType
from pyspark.sql.functions import col
import seaborn as sns
from io import BytesIO
from django.http import JsonResponse
from ipywidgets import interact, IntSlider

matplotlib.use('Agg')  # Use a non-interactive backend

def load_and_process_data(file_path="hdfs://localhost:9000/data.csv"):
    # Get Spark session
    spark = get_spark_session()

    # Step 1: Load the original CSV file from HDFS
    df_spark = spark.read.csv(file_path, header=True)

    # Step 2: Check for missing values
    essential_columns = ['etd_ticket_no', 'etd_amount', 'etd_kms', 'etd_from_stage_name', 'etd_to_stage_name']
    df_spark = df_spark.dropna(subset=essential_columns)

    # Step 3: Remove duplicates
    df_spark = df_spark.dropDuplicates()

    # Step 4: Convert string columns to appropriate types
    df_spark = df_spark.withColumn("etd_waybill_no", col("etd_waybill_no").cast(IntegerType())) \
                       .withColumn("etd_etm_no", col("etd_etm_no").cast(IntegerType())) \
                       .withColumn("etd_trip_no", col("etd_trip_no").cast(IntegerType())) \
                       .withColumn("etd_route_no", col("etd_route_no").cast(IntegerType())) \
                       .withColumn("etd_ticket_no", col("etd_ticket_no").cast(IntegerType())) \
                       .withColumn("etd_adult", col("etd_adult").cast(IntegerType())) \
                       .withColumn("etd_child", col("etd_child").cast(IntegerType())) \
                       .withColumn("etd_lugg", col("etd_lugg").cast(IntegerType())) \
                       .withColumn("etd_other", col("etd_other").cast(IntegerType())) \
                       .withColumn("etd_amount", col("etd_amount").cast(FloatType())) \
                       .withColumn("etd_rfid_slno", col("etd_rfid_slno").cast(IntegerType())) \
                       .withColumn("etd_rfid_cardno", col("etd_rfid_cardno").cast(IntegerType())) \
                       .withColumn("etd_rfid_bal_amt", col("etd_rfid_bal_amt").cast(FloatType())) \
                       .withColumn("etd_gender", col("etd_gender").cast(IntegerType())) \
                       .withColumn("etd_age", col("etd_age").cast(IntegerType())) \
                       .withColumn("etd_cash_or_card", col("etd_cash_or_card").cast(IntegerType())) \
                       .withColumn("etd_schd_no", col("etd_schd_no").cast(IntegerType())) \
                       .withColumn("etd_kms", col("etd_kms").cast(FloatType())) \
                       .withColumn("etd_route_kms", col("etd_route_kms").cast(IntegerType()))

    # Step 5: Filter out invalid data
    df_spark = df_spark.filter(col("etd_amount") >= 0).filter(col("etd_kms") >= 0)

    # Step 6: Standardize categorical values
    from pyspark.sql.functions import upper, trim

    df_spark = df_spark.withColumn("etd_from_stage_name", upper(trim(col("etd_from_stage_name")))) \
                       .withColumn("etd_to_stage_name", upper(trim(col("etd_to_stage_name"))))

    # Step 7: Save cleaned DataFrame back to CSV
    cleaned_file_path = "hdfs://localhost:9000/data_full4_cleaned.csv"
    df_spark.write.csv(cleaned_file_path, mode='overwrite', header=True)

    # Convert cleaned Spark DataFrame to Pandas DataFrame for further processing
    cleaned_df = df_spark.toPandas()

    return cleaned_df

# Calculate metrics like Load Factor, Passenger Kilometers, Capacity Kilometers, and Earning per km
def calculate_metrics(df):
    """Calculates Load Factor, Passenger Kilometers, Capacity Kilometers, and Earning per km."""
    
    # Example values (these need to be replaced by actual data from your dataset or user input)
    fare_per_km = 1.5  # Example: Fare per km (this could be dynamic)
    num_seats = 40     # Example: Number of seats in the bus

    # Aggregate revenue from passengers and kilometers
    df['total_passengers'] = df['etd_adult'] + df['etd_child']
    df['passenger_kilometers'] = df['etd_amount'] / fare_per_km  # Passenger km = Revenue from Passengers / Fare per km
    df['capacity_kilometers'] = df['etd_kms'] * num_seats        # Capacity km = Route length * Number of seats
    
    # Earning per km (Earning per km = Total earning / total effective km)
    total_earning = df['etd_amount'].sum()
    total_kms = df['etd_kms'].sum()
    earning_per_km = total_earning / total_kms if total_kms > 0 else 0
    
    # Load Factor = Passenger kilometers / Capacity kilometers
    total_passenger_kilometers = df['passenger_kilometers'].sum()
    total_capacity_kilometers = df['capacity_kilometers'].sum()
    load_factor = total_passenger_kilometers / total_capacity_kilometers if total_capacity_kilometers > 0 else 0

    return {
        'load_factor': load_factor,
        'passenger_kilometers': total_passenger_kilometers,
        'capacity_kilometers': total_capacity_kilometers,
        'earning_per_km': earning_per_km
    }

def generate_od_matrix_heatmap(df, start_row=0, end_row=20, start_col=0, end_col=20):
    # Create the OD matrix
    od_matrix = df.pivot_table(index='etd_from_stage_name', 
                               columns='etd_to_stage_name', 
                               values='etd_ticket_no', 
                               aggfunc='count', 
                               fill_value=0)

    # Normalize the values to range between 0 and 1 for better color scaling
    norm_od_matrix = (od_matrix - od_matrix.min().min()) / (od_matrix.max().max() - od_matrix.min().min())

    # Generate the heatmap image for the selected rows and columns
    plt.figure(figsize=(20, 10))
    sns.heatmap(norm_od_matrix.iloc[start_row:end_row, start_col:end_col], cmap='coolwarm', annot=False)
    plt.title('OD Matrix Heatmap')
    plt.ylabel('From Stage')
    plt.xlabel('To Stage')

    # Save the plot to a BytesIO object in PNG format
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode the image in Base64 format
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return image_base64
def index(request):
    # Load data from HDFS (or CSV)
    df = load_and_process_data()

    # Get unique route names for the dropdown
    unique_routes = df['etd_from_stage_name'].unique()

    # Calculate metrics
    metrics = calculate_metrics(df)

    # Default values for the slider ranges
    start_row = int(request.POST.get('start_row', 0))
    end_row = int(request.POST.get('end_row', 20))
    start_col = int(request.POST.get('start_col', 0))
    end_col = int(request.POST.get('end_col', 20))

    # Generate the heatmap image in Base64 format
    od_matrix_img = generate_od_matrix_heatmap(df, start_row, end_row, start_col, end_col)

    # Generate ARIMA forecast based on selected route
    route_name = request.POST.get('route_name', unique_routes[0])  # Default to the first route if none selected
    forecast_uri, error_message = generate_arima_forecast(route_name, df)

    # Render the template and pass the image and forecast data
    return render(request, 'myapp/index.html', {
        'od_matrix_img': od_matrix_img,
        'load_factor': metrics['load_factor'],
        'passenger_kilometers': metrics['passenger_kilometers'],
        'capacity_kilometers': metrics['capacity_kilometers'],
        'earning_per_km': metrics['earning_per_km'],
        'forecast_uri': forecast_uri,
        'error_message': error_message,
        'unique_routes': unique_routes,
    })

# Generate ARIMA forecast
def generate_arima_forecast(route_name, df):
    # Aggregate data by date and route
    df_aggregated = df.groupby(['etd_ticket_date', 'etd_route_name']).agg({
        'etd_adult': 'sum',
        'etd_child': 'sum',
        'etd_amount': 'sum'
    }).reset_index()
    
    # Calculate total passengers
    df_aggregated['total_passengers'] = df_aggregated['etd_adult'] + df_aggregated['etd_child']
    
    # Prepare time series for the specific route
    route_data = df_aggregated[df_aggregated['etd_route_name'] == route_name]
    
    # Set the index to 'etd_ticket_date' and ensure it is a DatetimeIndex
    route_data.set_index('etd_ticket_date', inplace=True)
    route_data.index = pd.to_datetime(route_data.index)

    # Set frequency to daily (or appropriate frequency)
    route_data = route_data['total_passengers'].asfreq('D', fill_value=0)
    
    if route_data.empty:
        return None, "No data available for the selected route."
    
    # Fit ARIMA model
    p, d, q = 1, 1, 1  # You can tune these values
    model = ARIMA(route_data, order=(p, d, q))
    
    # Fit the model
    model_fit = model.fit()

    # Forecast
    forecast_steps = 30  # Forecast for the next 30 days
    forecast = model_fit.forecast(steps=forecast_steps)
    
    # Create a date range for the forecast
    forecast_dates = pd.date_range(start=route_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(route_data, label='Historical Data')
    ax.plot(forecast_dates, forecast, label='Forecast', color='red')
    ax.set_xlim(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_title(f'Passenger Forecast for {route_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Passengers')
    ax.legend()
    ax.grid(True)
    
    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    
    return uri, None
