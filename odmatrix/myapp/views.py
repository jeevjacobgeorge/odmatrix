import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import folium
from geopy.geocoders import Nominatim
from django.shortcuts import render
from io import BytesIO
import base64
from .models import ODMatrix
import matplotlib
matplotlib.use('Agg')

# Function to generate Heatmap and return as a base64 string
# Function to generate Heatmap and return as a base64 string
def generate_heatmap(od_matrix):
    plt.figure(figsize=(12, 8))
    
    # Ensure columns are correctly labeled for the pivot
    od_matrix_pivot = od_matrix.pivot_table(index="from_stage", columns="to_stage", values="passenger_count", fill_value=0)
    
    sns.heatmap(od_matrix_pivot, cmap='YlGnBu', linewidths=0.5)
    plt.title('OD Matrix Heatmap (Passenger Flows)', fontsize=16)
    plt.xlabel('Destination', fontsize=12)
    plt.ylabel('Origin', fontsize=12)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    return base64.b64encode(image_png).decode('utf-8')


# Function to generate Sankey Diagram
def generate_sankey(od_matrix):
    nodes = list(set(od_matrix['from_stage']).union(set(od_matrix['to_stage'])))
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    sources = [node_indices[from_stage] for from_stage in od_matrix['from_stage']]
    targets = [node_indices[to_stage] for to_stage in od_matrix['to_stage']]
    values = od_matrix['passenger_count']

    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=nodes),
        link=dict(source=sources, target=targets, value=values)
    ))

    fig.update_layout(title_text="Passenger Flow Sankey Diagram", font_size=12)
    return fig.to_html(full_html=False)

# Function to generate Flow Map
def generate_flow_map(od_matrix):
    geolocator = Nominatim(user_agent="geoapi")
    
    def geocode(stage_name):
        try:
            location = geolocator.geocode(stage_name)
            return location.latitude, location.longitude
        except:
            return None, None

    od_matrix['from_latlon'] = od_matrix['from_stage'].apply(geocode)
    od_matrix['to_latlon'] = od_matrix['to_stage'].apply(geocode)
    
    m = folium.Map(location=[10.0, 76.0], zoom_start=7)
    for _, row in od_matrix.iterrows():
        from_lat, from_lon = row['from_latlon']
        to_lat, to_lon = row['to_latlon']
        if from_lat and to_lat:
            folium.PolyLine([(from_lat, from_lon), (to_lat, to_lon)], weight=row['passenger_count']/10, color='blue').add_to(m)
    
    return m._repr_html_()

# Function to process data in chunks and save the OD matrix into the database
def process_large_data_in_chunks(file_path, chunk_size=10000):
    od_matrix = pd.DataFrame()

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        # Group by 'etd_from_stage_name' and 'etd_to_stage_name' in the chunk
        chunk_od = chunk.groupby(['etd_from_stage_name', 'etd_to_stage_name']).size().reset_index(name='passenger_count')

        if od_matrix.empty:
            od_matrix = chunk_od
        else:
            od_matrix = od_matrix.merge(chunk_od, on=['etd_from_stage_name', 'etd_to_stage_name'], how='outer', suffixes=('_old', '_new'))
            od_matrix['passenger_count'] = od_matrix['passenger_count_old'].fillna(0) + od_matrix['passenger_count_new'].fillna(0)
            od_matrix.drop(['passenger_count_old', 'passenger_count_new'], axis=1, inplace=True)

    for _, row in od_matrix.iterrows():
        ODMatrix.objects.create(
            from_stage=row['etd_from_stage_name'],
            to_stage=row['etd_to_stage_name'],
            passenger_count=row['passenger_count']
        )

# Function to fetch and visualize the OD matrix
def visualize_od_matrix(request):
    if ODMatrix.objects.exists():
        od_matrix = pd.DataFrame(list(ODMatrix.objects.values('from_stage', 'to_stage', 'passenger_count')))
    else:
        file_path = 'path_to_your_large_data_file.csv'
        process_large_data_in_chunks(file_path)

        od_matrix = pd.DataFrame(list(ODMatrix.objects.values('from_stage', 'to_stage', 'passenger_count')))

    heatmap_img = generate_heatmap(od_matrix)
    sankey_diagram = generate_sankey(od_matrix)
    flow_map = generate_flow_map(od_matrix)

    context = {
        'heatmap_img': heatmap_img,
        'sankey_diagram': sankey_diagram,
        'flow_map': flow_map
    }

    return render(request, 'visualize_od_matrix.html', context)
