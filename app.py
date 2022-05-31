from pathlib import Path
from fastai.vision.all import *
from fastai.vision.widgets import *

import streamlit as st
import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

plt =platform.system()
if plt == "Windows": pathlib.WindowsPath = pathlib.PosixPath


st.header("Image regression")


# Image centre function
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1, c2])

# Make prediction


def predictResized2(image):
    #image = PILImage.create(image)
    image = PILImage(image.resize((320, 240)))
    results = learn_inf.predict(image)
    st.write(results)
    point = results[0][0]
    plt.scatter(point[0], point[1], zorder=2)
    plt.imshow(image, zorder=1)
    # allows us to make pyplot() without figue attribute
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


# Load Model #
path = Path()
learn_inf = load_learner(path/'Image_center_regress2.pkl')

# File uploader #
try:
    uploaded_file = st.file_uploader(
        "Upload Files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        image_file = PILImage.create((uploaded_file))

# Display uploaded image #
    st.image(image_file.to_thumb(400, 400), caption='Uploaded Image')

    # make prediction and display result #
    if st.button('Find Centre'):
        predictResized2(image_file)
    else:
        st.write(f'Click the button to Find Centre')

except NameError:
    # error will only be visible in the comand line #
    print("Name error is caught")
