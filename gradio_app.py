
import gradio as gr 

from theme_classifier import ThemeClassifier
from character_network import EntityClassifier, NetworkGenerator
from text_classification import JutsuClassifier
from dotenv import load_dotenv
load_dotenv()
import os


def get_themes(themes, subtitles_path, save_path):

    ''' '''

    theme_list = themes.split(',')
    theme_classifer = ThemeClassifier(theme_list)

    themes_df = theme_classifer.get_themes(dataset_path=subtitles_path, save_path=save_path)

    # format dataframe. 
    theme_list = [ theme for theme in theme_list if theme != 'dialogue' ]
    themes_df = themes_df[theme_list]

    themes_df = themes_df[theme_list].sum().reset_index()
    themes_df.columns = ['Theme', 'Score']

    output_chart = gr.BarPlot(
        themes_df,
        x='Theme', 
        y='Score',
        title='Series Themes',
        tooltip=['Theme', 'Score'], 
        vertical=False, 
        width=500,
        height=260
    )

    return output_chart


def retrieve_character_network(subtitles_path, entity_path):
    
    ''' '''

    entity_classifier = EntityClassifier()
    entity_dataframe = entity_classifier.get_classes(subtitles_path, entity_path)

    charater_network = NetworkGenerator()
    relationships_dataframe = charater_network.generate_character_network(entity_dataframe)

    html = charater_network.draw_network_graph(relationships_dataframe)

    return html
    

def classify_text(text_classification_model, data_path, input_data):

    classifier = JutsuClassifier(model_path=text_classification_model, data_path=data_path, huggingface_token=os.getenv('HUGGING_FACE_TOKEN'))

    output = classifier.classify_jutsu(input_data)

    return output


def main():

    with gr.Blocks() as iface:

        ''' Zero Shot Theme Classification. '''

        with gr.Row():
            with gr.Column():
                gr.HTML('<h1>Theme Classification (Zero Shot Classifiers)</h1>')
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label='Themes')
                        subtitles_path = gr.Textbox(label='Subtitles or Script Path')
                        save_path = gr.Textbox(label='Save Path')
                        get_themes_btn = gr.Button('Get Themes')

                        get_themes_btn.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])

        ''' Character Network Generator Graph. '''

        with gr.Row():
            with gr.Column():
                gr.HTML('<h1>Character Network (Entity Graphs)</h1>')
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label='Subtitles Script Path.')
                        entity_path = gr.Textbox(label='Entity Save Path.')
                        retrieve_graph_btn = gr.Button('Retrieve Character Network.')
                        retrieve_graph_btn.click(retrieve_character_network, inputs=[subtitles_path, entity_path], outputs=[network_html])

        ''' LLM Text Classification. '''

        with gr.Row():
            with gr.Column():
                gr.HTML('<h1>Text Classification (Large Language Models).</h1>')
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label='Classification Output')
                    with gr.Column():
                        text_classification_model = gr.Textbox(label='Model Path')
                        text_classification_data_path = gr.Textbox(label='Data Path')
                        input_text = gr.Textbox(label='Text Input')
                        classify_input_btn = gr.Button('Classifiy Text (Jutsu)')
                        classify_input_btn.click(classify_text, inputs=[text_classification_model, text_classification_data_path, input_text], outputs=[text_classification_output])

    iface.launch(share=True)


if __name__ == '__main__':
    main()
