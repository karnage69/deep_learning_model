import gradio as gr
from cnn_dep import predict


interface = gr.Interface(
    fn=predict,

    inputs=gr.Image(
        type="pil",
        label="Upload Image"
    ),

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="Image Classifier",

    description="Upload an image and get prediction"
)


interface.launch()