import streamlit as st
import torch
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

from cnn import CNN


device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()


st.title("Digit Drawing ")
st.write("Draw a digit:")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Stroke fill (inside shapes)
    stroke_width=10,                      # Thickness of your pen
    stroke_color="white",                 # What you're drawing with
    background_color="black",            # The canvas background (invisible space = 0)
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data).astype(np.uint8)).convert("L")


    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_digit = torch.argmax(output, dim=1).item()


    st.markdown(f"Model read: **{predicted_digit}**")

    
