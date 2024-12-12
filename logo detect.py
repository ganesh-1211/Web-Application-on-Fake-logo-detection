import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure, feature
import numpy as np
import cv2 as cv
import glob
import os

def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    md = np.median(gray)
    sigma = 0.35
    low = int(max(0, (1.0 - sigma) * md))
    up = int(min(255, (1.0 + sigma) * md))
    edged = cv.Canny(gray, low, up)
    (x, y, w, h) = cv.boundingRect(edged)
    logo = gray[y:y + h, x:x + w]
    logo = cv.resize(logo, (200, 100))
    return logo

def extract_hog_features(logo):
    hist = feature.hog(
        logo,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L1"
    )
    return hist

# Get Paths
trainingPath = r'C:\Users\Ganes\OneDrive\Desktop\Fake logo detection\train'
# Init Lists
hists = []  # histogram of Image
labels = []  # Label of Image

for imagePath in glob.glob(trainingPath + "/*.*"):
    label = os.path.splitext(os.path.basename(imagePath))[0]
    image = cv.imread(imagePath)
    try:
        logo = preprocess_image(image)
        hist = extract_hog_features(logo)
        hists.append(hist)
        labels.append(label)
    except cv.error:
        print(imagePath)
        print("Training Image couldn't be read")

model = KNeighborsClassifier(n_neighbors=1)
model.fit(hists, labels)

st.title("Fake Logo Detection")
st.write("This application helps to detect fake and original logos")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = cv.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv.IMREAD_UNCHANGED)
    try:
        logo = preprocess_image(image)
        hist = extract_hog_features(logo)

        predict = model.predict(hist.reshape(1, -1))[0]

        st.image(image,use_column_width=True)
        dpath=r'C:\Users\Ganes\OneDrive\Desktop\Fake logo detection\Genuine'
        filepath=os.path.join(dpath,str(predict)+".jpg")
        if os.path.isfile(filepath):
            st.write("Original")
        else:
            st.write("Fake")

    except cv.error:
        st.write("Error: Image couldn't be processed")
