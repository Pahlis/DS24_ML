import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Konfigurera sidan
st.set_page_config(page_title="MNIST Sifferklassificerare", layout="wide")

st.markdown("""
    <style>
    body, p, div {
        font-size: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Titel
st.title("MNIST Sifferklassificerare")
st.write("Välj om du vill rita en siffra i rutan eller ladda upp en bild för att klassificera.")



@st.cache_resource
def load_model():
    try:
        model = joblib.load("mnist_best_model.joblib")
        norm_info = joblib.load("mnist_normalization.joblib")
        return model, norm_info
    except FileNotFoundError:
        st.error("Kunde inte hitta modell eller normaliseringsfiler. Kontrollera att de har sparats.")
        return None, None


model, norm_info = load_model()


if model is not None:
    st.write("Använder modell: MLPC (Neuralt Nätverk)")

    # Skapa två flikar istället för kolumner
    tab1, tab2 = st.tabs(["Rita en siffra", "Ladda upp en bild"])

    with tab1:
        st.subheader("Rita en siffra")
        from streamlit_drawable_canvas import st_canvas

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("Klassificera ritad siffra"):
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                img = img.convert('L')
                img = img.resize((28, 28))

                img_array = np.array(img)
                if np.mean(img_array) > 128:
                    img_array = 255 - img_array

                img_array = img_array / norm_info["scale_factor"]

                st.image(img_array, caption="Förbehandlad bild (28x28)", width=150)

                prediction = model.predict(img_array.reshape(1, -1))
                st.success(f"Klassificering: {prediction[0]}")

    with tab2:
        st.subheader("Ladda upp en bild")
        uploaded_file = st.file_uploader("Välj en bildfil", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uppladdad bild", width=300)

            img = image.convert('L')
            img = img.resize((28, 28))

            img_array = np.array(img)
            if np.mean(img_array) > 128:
                img_array = 255 - img_array

            img_array = img_array / norm_info["scale_factor"]

            st.image(img_array, caption="Förbehandlad bild (28x28)", width=150)

            prediction = model.predict(img_array.reshape(1, -1))
            st.success(f"Klassificering: {prediction[0]}")

    st.markdown("---")
    st.subheader("Om appen")
    st.markdown("""
    Denna app använder en maskininlärningsmodell tränad på MNIST-datan för att klassificera handskrivna siffror.
    Modellen är tränad med hjälp av scikit-learn och sparad med joblib.

    - **Dataset:** MNIST (70,000 handskrivna siffror, 28x28 pixlar)
    - **Modell:** MLPC (Neuralt Nätverk)
    - **Accuracy på träningsdata:** Ungefär 98%
    - **Accuracy på testdata:** Ungefär 98%
    """)
else:
    st.error("Ingen modell laddad. Se till att köra träningsskriptet först.")
