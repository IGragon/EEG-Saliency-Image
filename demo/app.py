import torch
from demo.model import SaliencyEEGGuidedDiffusion
from pathlib import Path
from PIL import Image
import streamlit as st

VISUAL_STIMULI_PATH = Path(
    "/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/data/images/test_images"
)
SALIENCY_MAPS_PATH = Path(
    "/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/data/images/test_images_saliency_maps"
)
EEG_EMBEDDINGS = torch.load(
    "/home/igragon/Projects/innopolis_thesis_2025/EEG-Salience-Image/data/emb_eeg/ATM_S_eeg_features_sub-08_test.pt"
)

IMAGE_SIZE = (512, 512)

VISUAL_STIMULI = [
    f"{img.parent.name}/{img.name}"
    for img in sorted(VISUAL_STIMULI_PATH.glob("*/*.jpg"))
]


@st.cache_resource
def load_model():
    st.spinner("Loading model...")
    model = SaliencyEEGGuidedDiffusion()
    st.success("Model loaded")

    return model


# App layout
st.title("Saliency-guided image reconstruction from EEG signals")
st.write("Select visual stimuli and saliency map to reconstruct the image")

model = load_model()

vis_stim_col, sal_col, result_col = st.columns(3)
vis_stim_eeg_emb = None
sal_map = None

with vis_stim_col:
    st.write("Visual stimuli")
    stimuli_path = st.selectbox(
        "Visual stimuli",
        VISUAL_STIMULI,
        placeholder="Select visual stimuli",
    )
    if stimuli_path:
        stimuli_index = VISUAL_STIMULI.index(stimuli_path)
        vis_stim = VISUAL_STIMULI_PATH / stimuli_path
        vis_stim_eeg_emb = EEG_EMBEDDINGS[[stimuli_index]].unsqueeze(0)
        st.image(vis_stim)

with sal_col:
    st.write("Saliency map")
    st.checkbox("Load custom saliency map", key="is_load_custom_saliency_map")
    if stimuli_path:
        if st.session_state.is_load_custom_saliency_map:
            sal_map = st.file_uploader(
                "Upload saliency map",
                type=["png", "jpg", "jpeg"],
            )
        else:
            sal_map = SALIENCY_MAPS_PATH / stimuli_path
        if sal_map:
            sal_map = Image.open(sal_map).resize(IMAGE_SIZE)
            st.image(sal_map)

with result_col:
    st.write("Reconstructed image")
    if vis_stim_eeg_emb is not None and sal_map is not None:
        if st.button("Reconstruct image"):
            with st.spinner("Reconstructing image..."):
                image = model.process(vis_stim_eeg_emb, [sal_map])
                st.image(image)
