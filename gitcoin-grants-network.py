import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import plotly.graph_objs as go
import plotly.express as px
import locale
import networkx as nx
import time

BASE_URL = "https://indexer-grants-stack.gitcoin.co/data"
time_to_live = 900  # 15 minutes

st.set_page_config(
    page_title="Gitcoin Grants Networks",
    page_icon="📊",
    layout="wide",
)

st.title('Gitcoin Grants')
st.write('The Gitcoin Grants Program is a quarterly initiative that empowers everyday believers to drive funding toward what they believe matters, with the impact of individual donations being magnified by the use of the [Quadratic Funding (QF)](https://wtfisqf.com) distribution mechanism.')
st.write('This network graph is still in development. It helps visualize the connections between donors and projects in the Gitcoin Grants Beta Rounds. The graph is interactive, so you can hover over a node to see who it is, zoom in and out and drag the graph around to explore it.')
st.write('One use for this graph is to identify interesting outliers such as grants who have their own distinct donor base.')

def safe_get(data, *keys):
    """Safely retrieve nested dictionary keys."""
    temp = data
    for key in keys:
        if isinstance(temp, dict) and key in temp:
            temp = temp[key]
        else:
            return None
    return temp

def load_data_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.RequestException as e:
        st.warning(f"Failed to fetch data from {url}. Error: {e}")
        return []

@st.cache_data(ttl=time_to_live)
def load_data(chain_id, round_id, data_type):
    url = f"{BASE_URL}/{chain_id}/rounds/{round_id}/{data_type}.json"
    return load_data_from_url(url)

def transform_projects_data(data):
    projects = []
    for project in data:
        title = safe_get(project, 'metadata', 'application', 'project', 'title')
        grantAddress = safe_get(project, 'metadata', 'application', 'recipient')
        description = safe_get(project, 'metadata', 'application', 'project', 'description')
        
        if title and grantAddress:  # Ensure required fields are available
            project_data = {
                'projectId': project['projectId'],
                'title': title,
                'grantAddress': grantAddress,
                'status': project['status'],
                'amountUSD': project['amountUSD'],
                'votes': project['votes'],
                'uniqueContributors': project['uniqueContributors'],
                'description': description
            }
            projects.append(project_data)
    return projects

@st.cache_data(ttl=time_to_live)
def load_passport_data():
    url = f"{BASE_URL}/passport_scores.json"
    data = load_data_from_url(url)
    
    passports = []
    for passport in data:
        address = passport.get('address')
        last_score_timestamp = passport.get('last_score_timestamp')
        status = passport.get('status')
        rawScore = safe_get(passport, 'evidence', 'rawScore') or 0

        if address:  # Ensure the required field is available
            passport_data = {
                'address': address,
                'last_score_timestamp': last_score_timestamp,
                'status': status,
                'rawScore': rawScore
            }
            passports.append(passport_data)

    df = pd.DataFrame(passports)
    #df['last_score_timestamp'] = pd.to_datetime(df['last_score_timestamp'])
    return df

def compute_timestamp(row, starting_time, chain_starting_blocks):
    # Get the starting block for the chain_id
    starting_block = chain_starting_blocks[row['chain_id']]
    # Calculate the timestamp based on the blockNumber and starting block
    timestamp = starting_time + pd.to_timedelta((row['blockNumber'] - starting_block) * 2, unit='s')
    return timestamp

# Usage
data_load_state = st.text('Loading data...')
round_data = pd.read_csv('gg18_rounds.csv')

dfv_list = []
dfp_list = []
for _, row in round_data.iterrows():
    raw_projects_data = load_data(str(row['chain_id']), str(row['round_id']), "applications")
    projects_list = transform_projects_data(raw_projects_data)
    dfp = pd.DataFrame(projects_list)
    dfv = pd.DataFrame(load_data(str(row['chain_id']), str(row['round_id']), "votes"))

    dfp['round_id'] = row['round_id']
    dfp['chain_id'] = row['chain_id']
    dfp['round_name'] = row['round_name']
    
    dfv['round_id'] = row['round_id']
    dfv['chain_id'] = row['chain_id']
    dfv['round_name'] = row['round_name']

    dfv_list.append(dfv)
    dfp_list.append(dfp)

dfv = pd.concat(dfv_list)
dfp = pd.concat(dfp_list)

token_map = {
    "0x0000000000000000000000000000000000000000": "ETH",
    "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1": "DAI",
}
dfv["token_symbol"] = dfv["token"].map(token_map)

chain_starting_blocks = dfv.groupby('chain_id')['blockNumber'].min().to_dict()
starting_time = pd.to_datetime('2023/08/15 12:00 PM UTC')
dfv['timestamp'] = dfv.apply(compute_timestamp, args=(starting_time, chain_starting_blocks), axis=1)

dfpp = load_passport_data()



data_load_state.text("")

# selectbox to select the round
option = st.selectbox(
    'Select Round',
    dfv['round_name'].unique(), index=3)

dfv = dfv[dfv['round_name'] == option]
dfp = dfp[dfp['round_name'] == option]
round_data = round_data[round_data['round_name'] == option]
dfv = pd.merge(dfv, dfp[['projectId', 'title']], how='left', left_on='projectId', right_on='projectId')
dfv = pd.merge(dfv, dfpp[['address', 'rawScore']], how='left', left_on='voter', right_on='address')
dfv['rawScore'] = dfv['rawScore'].fillna(0)

# sum amountUSD group by voter and grantAddress
dfv = dfv.groupby(['voter', 'grantAddress', 'title']).agg({'amountUSD': 'sum', 'timestamp': 'min', 'rawScore':'max'}).reset_index()

# Minimum donation amount to include, start at 10
min_donation = st.slider('Minimum donation amount', value=10, max_value=50, min_value=1, step=1)
# Minimum passport score to include, start at 20
min_passport_score = st.slider('Minimum Passport Score', value=20, max_value=100, min_value=1, step=1)

# Filter the dataframe to include only rows with donation amounts above the threshold
dfv = dfv[dfv['amountUSD'] > min_donation]
# Filter the dataframe to include only rows with donation amounts above the threshold
df = dfv[dfv['rawScore'] > min_passport_score]


count_connections = dfv.shape[0]
count_voters = dfv['voter'].nunique()
count_grants = dfv['title'].nunique()



color_toggle = st.checkbox('Toggle colors', value=True)

if color_toggle:
    grants_color = '#FF7043'
    grantee_color_string = 'orange'
    voters_color = '#B3DE9F'
    voter_color_string = 'green'
    line_color = '#6E9A82'
else:
    grants_color = 'blue'
    grantee_color_string = 'blue'
    voters_color = 'red'
    voter_color_string = 'red'
    line_color = '#008F11'

note_string = '**- Note: ' + str(count_grants) + ' Grantees are in ' + grantee_color_string + ' and ' + str(count_voters) + ' donors/voters are in ' + voter_color_string + ' forming ' + str(count_connections) + ' connections.**'
st.markdown(note_string)
st.markdown('**- Tip: Go fullscreen with the arrows in the top-right for a better view.**')
# Initialize a new Graph
B = nx.Graph()

# Create nodes with the bipartite attribute
B.add_nodes_from(dfv['voter'].unique(), bipartite=0, color=voters_color) 
B.add_nodes_from(dfv['title'].unique(), bipartite=1, color=grants_color) 



# Add edges with amountUSD as an attribute
for _, row in dfv.iterrows():
    B.add_edge(row['voter'], row['title'], amountUSD=row['amountUSD'])



# Compute the layout
current_time = time.time()
pos = nx.spring_layout(B, dim=3, k = .09, iterations=50)
new_time = time.time()


    
# Extract node information
node_x = [coord[0] for coord in pos.values()]
node_y = [coord[1] for coord in pos.values()]
node_z = [coord[2] for coord in pos.values()] # added z-coordinates for 3D
node_names = list(pos.keys())
# Compute the degrees of the nodes 
degrees = np.array([B.degree(node_name) for node_name in node_names])
# Apply the natural logarithm to the degrees 
log_degrees = np.log(degrees + 1)
# Min-Max scaling manually
#min_size = 10  # minimum size
#max_size = 50  # maximum size
#node_sizes = ((log_degrees - np.min(log_degrees)) / (np.max(log_degrees) - np.min(log_degrees))) * (max_size - min_size) + min_size
node_sizes = log_degrees * 10

# Extract edge information
edge_x = []
edge_y = []
edge_z = []  
edge_weights = []

for edge in B.edges(data=True):
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])  
    edge_weights.append(edge[2]['amountUSD'])

# Create the edge traces
edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z, 
    line=dict(width=1, color=line_color),
    hoverinfo='none',
    mode='lines',
    marker=dict(opacity=0.5))


# Create the node traces
node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        color=[data['color'] for _, data in B.nodes(data=True)],  # color is now assigned based on node data
        size=node_sizes,
        opacity=1,
        sizemode='diameter'
    ))


node_adjacencies = []
for node, adjacencies in enumerate(B.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
node_trace.marker.color = [data[1]['color'] for data in B.nodes(data=True)]


# Prepare text information for hovering
node_trace.text = [f'{name}: {adj} connections' for name, adj in zip(node_names, node_adjacencies)]

# Create the figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='3D Network graph of voters and grants',
                    titlefont=dict(size=20),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        text="This graph shows the connections between voters and grants based on donation data.",
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002 )],
                    scene = dict(
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        zaxis_title='Z Axis')))
                        
st.plotly_chart(fig, use_container_width=True)
st.caption('Time to compute layout: ' + str(round(new_time - current_time, 2)) + ' seconds')