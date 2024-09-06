import matplotlib.pyplot as plt 
import seaborn as sns 
import gradio as gr 

from theme_classifier import ThemeClassifier


def get_themes(themes, subtitles_path, save_path):

    ''' '''

    theme_list = themes.split(',')
    theme_classifer = ThemeClassifier(theme_list)

    themes_df = theme_classifer.get_themes(dataset_path=subtitles_path, save_path=save_path)

    # format dataframe. 
    theme_list = [ theme for theme in theme_list if theme != 'dialogue' ]
    themes_df = themes_df[theme_list]

    themes_df = themes_df[theme_list].sum().rest_index()
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
    

def main():

    print('app loading!')

    with gr.Blocks() as iface:
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

    iface.launch(share=True)


if __name__ == '__main__':

    main()
