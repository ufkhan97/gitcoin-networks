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


st.set_page_config(
    page_title="Gitcoin Beta Rounds",
    page_icon="📊",
    layout="wide",

)

st.title('Gitcoin Beta Rounds')
st.write('The Gitcoin Grants Program is a quarterly initiative that empowers everyday believers to drive funding toward what they believe matters, with the impact of individual donations being magnified by the use of the [Quadratic Funding (QF)](https://wtfisqf.com) distribution mechanism.')
st.write('')
st.write('This network graph is still in development. It helps visualize the connections between donors and projects in the Gitcoin Grants Beta Rounds. The graph is interactive, so you can hover over a node to see who it is, zoom in and out and drag the graph around to explore it.')
st.write('One use for this graph is to identify interesting outliers such as grants who have their own distinct donor base.')


chain_id = '1'


@st.cache_data(ttl=3000)
def load_chain_data(chain_id):
    chain_url = 'https://indexer-grants-stack.gitcoin.co/data/' + chain_id + '/rounds.json'
    try:
        response = requests.get(chain_url)
        if response.status_code == 200:
            chain_data = response.json()
            rounds = []
            for round in chain_data:
                if round['metadata'] is not None:
                    round_data = {
                        'round_id': round['id'],
                        'name': round['metadata']['name'],
                        'amountUSD': round['amountUSD'],
                        'votes': round['votes'],
                        'description': round['metadata']['description'] if 'description' in round['metadata'] else '',
                        'matchingFundsAvailable': round['metadata']['matchingFunds']['matchingFundsAvailable'] if 'matchingFunds' in round['metadata'] else '',
                        'matchingCap': round['metadata']['matchingFunds']['matchingCap'] if 'matchingFunds' in round['metadata'] else '',
                        'roundStartTime': datetime.datetime.utcfromtimestamp(int(round['roundStartTime'])), # create a datetime object from the timestamp in UTC time
                        'roundEndTime': datetime.datetime.utcfromtimestamp(int(round['roundEndTime']))
                    }
                    rounds.append(round_data)
            df = pd.DataFrame(rounds)
            # Filter to beta rounds
            start_time = datetime.datetime(2023, 4, 26, 15, 0, 0)
            end_time = datetime.datetime(2023, 5, 9, 23, 59, 0)
            # filter to only include rounds with votes > 0 and roundStartTime <= start_time and roundEndTime == end_time
            df = df[(df['votes'] > 0) & (df['roundStartTime'] <= start_time) & (df['roundEndTime'] == end_time)]
            return df 
    except: 
        return pd.DataFrame()

@st.cache_data(ttl=3000)
def load_round_projects_data(round_id):
    # prepare the URLs
    projects_url = 'https://indexer-grants-stack.gitcoin.co/data/1/rounds/' + round_id + '/projects.json'
    
    try:
        # download the Projects JSON data from the URL
        response = requests.get(projects_url)
        if response.status_code == 200:
            projects_data = response.json()

        # Extract the relevant data from each project
        projects = []
        for project in projects_data:
            project_data = {
                'id': project['id'],
                'title': project['metadata']['application']['project']['title'],
                'description': project['metadata']['application']['project']['description'],
                'status': project['status'],
                'amountUSD': project['amountUSD'],
                'votes': project['votes'],
                'uniqueContributors': project['uniqueContributors']
            }
            projects.append(project_data)
        # Create a DataFrame from the extracted data
        dfp = pd.DataFrame(projects)
        # Reorder the columns to match the desired order and rename column id to project_id
        dfp = dfp[['id', 'title', 'description', 'status', 'amountUSD', 'votes', 'uniqueContributors']]
        dfp = dfp.rename(columns={'id': 'project_id'})
        # Filter to only approved projects
        dfp = dfp[dfp['status'] == 'APPROVED']
        return dfp
    except:
        return pd.DataFrame()
    
@st.cache_data(ttl=3000)
def load_round_votes_data(round_id):
    votes_url = 'https://indexer-grants-stack.gitcoin.co/data/1/rounds/' + round_id + '/votes.json'
    try:
        # download the Votes JSON data from the URL
        response = requests.get(votes_url)
        if response.status_code == 200:
            votes_data = response.json()
        df = pd.DataFrame(votes_data)
        return df
    except:
        return pd.DataFrame()


data_load_state = st.text('Loading data...')
chain_data = load_chain_data(chain_id)
data_load_state.text("")

# selectbox to select the round
option = st.selectbox(
    'Select Round',
    chain_data['name'], index=3)



data_load_state = st.text('Loading data...')
# load round data for the option selected by looking up the round id with that name in the chain_data df
round_id = chain_data[chain_data['name'] == option]['round_id'].values[0]
dfp = load_round_projects_data(round_id)
dfv = load_round_votes_data(round_id)
data_load_state.text("")

dfv = pd.merge(dfv, dfp[['project_id', 'title', 'status']], how='left', left_on='projectId', right_on='project_id')



# sum amountUSD group by voter and grantAddress
dfv = dfv.groupby(['voter', 'grantAddress', 'title', 'status']).agg({'amountUSD': 'sum'}).reset_index()

# Minimum donation amount to include, start at 10
min_donation = st.slider('Minimum donation amount', value=10, max_value=1000, min_value=1, step=1)

# Filter the dataframe to include only rows with donation amounts above the threshold
dfv = dfv[dfv['amountUSD'] > min_donation]
# st.write(dfv)

# count the number of rows, unique voters, and unique grant addresses


# make three columns in one row for metrics

count_connections = dfv.shape[0]
count_voters = dfv['voter'].nunique()
count_grants = dfv['title'].nunique()
#col1, col2, col3 = st.columns(3)
#col1 = st.metric(label="Connections", value=dfv.shape[0])
#col2 = st.metric(label="Voters", value=dfv['voter'].nunique())
#col3 = st.metric(label="Grants", value=dfv['title'].nunique())


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
if count_connections > 5000:
    pos = nx.spring_layout(B, dim=3, k = .09, iterations=50)
else:
    pos = nx.spring_layout(B, dim=3, k = .09, iterations=100)
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