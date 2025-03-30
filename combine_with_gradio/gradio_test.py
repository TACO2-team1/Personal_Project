import gradio as gr
from face_alignment_master import model2_facealign

iface = gr.Interface(
    fn=model2_facealign.s3fd_pgd_attack,
    inputs=[
        gr.Image(label="Upload Image", type="filepath"),  
        gr.Slider(0.5, 0.7, step=0.01, label="EPS"),  
        gr.Slider(0.1, 1, step=0.1, label="ALPHA"),  
        gr.Slider(1, 15, step=1, label="ATTACK_STEPS"), 
    ],
    outputs=gr.Image(label="Adversarial Image"), 
    title="PGD Attack with SFD Detector",
    description="Upload an image and adjust EPS, ALPHA, and ATTACK_STEPS to see the adversarial effect on the image."
)

iface.launch(share=False)

