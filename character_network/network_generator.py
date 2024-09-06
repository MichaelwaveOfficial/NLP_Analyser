import pandas as pd 
from pyvis.network import Network
import networkx as nx


class NetworkGenerator(object):

    
    def __init__(self) -> None:
        pass 


    def generate_character_network(self, dataframe):

        ''' '''

        window_size=10
        entity_relationships = []

        for row in dataframe['entities']:

            previous_entities_window = []

            for sentence in row:
                previous_entities_window.append(list(sentence))
                previous_entities_window = previous_entities_window[-window_size:]

                # Flatten 2D to 1D list. 
                previous_entities_flattened = sum(previous_entities_window, [])

                for entity in sentence:

                    for windowed_entity in previous_entities_flattened:

                        if entity != windowed_entity:
                            entity_relationships.append(sorted([entity, windowed_entity]))

        relationship_dataframe = pd.DataFrame({'value' : entity_relationships})

        relationship_dataframe['source'] = relationship_dataframe['value'].apply(lambda x : x[0])
        relationship_dataframe['target'] = relationship_dataframe['value'].apply(lambda x : x[1])
        relationship_dataframe = relationship_dataframe.groupby(['source', 'target']).count().reset_index()
        relationship_dataframe = relationship_dataframe.sort_values('value', ascending=False)

        return relationship_dataframe
    

    def draw_network_graph(self, relationship_dataframe):

        relationship_dataframe = relationship_dataframe.sort_values('value', ascending=False)
        relationship_dataframe = relationship_dataframe.head(200)

        Graph = nx.from_pandas_edgelist(relationship_dataframe, source='source', target='target', edge_attr='value', create_using=nx.Graph())

        net = Network(notebook=True, height='750px', width='1000px', bgcolor='#222222', font_color='white', cdn_resources='remote')

        node_degree = dict(Graph.degree)

        nx.set_node_attributes(Graph, node_degree, 'size')
        net.from_nx(Graph)
        
        html = net.generate_html()
        html = html.replace("'", "\"")

        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
            display-capture; encrypted-media;" sandbox="allow-modals allow-forms
            allow-scripts allow-same-origin allow-popups
            allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
            allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
        
        return output_html