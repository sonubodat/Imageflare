import streamlit as st
from PIL import Image, ImageOps
import io

# Import all our logic functions from the src package
from src import enhancements, noise, denoise, filters, analysis, utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Flare",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.history = []
    st.session_state.history_index = -1

def add_to_history(image):
    """Adds a new image state to the history for undo/redo."""
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    st.session_state.history.append(image.copy())
    st.session_state.history_index += 1

def apply_operation(operation_func, **kwargs):
    """Generic function to apply an operation and update history."""
    if st.session_state.processed_image:
        st.session_state.processed_image = operation_func(st.session_state.processed_image, **kwargs)
        add_to_history(st.session_state.processed_image)
        st.rerun()

# --- Sidebar ---
st.sidebar.title("✨ Image Flare")

# File Uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file:
    if st.session_state.get('uploaded_file_name') != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.session_state.original_image = image.copy()
        st.session_state.processed_image = image.copy()
        st.session_state.history = [image.copy()]
        st.session_state.history_index = 0

if st.session_state.original_image:
    st.sidebar.header("Operations")
    
    # Noise, Denoise, and Filter sections...
    with st.sidebar.expander("Noise Operations"):
        noise_ops = ["Gaussian", "Rayleigh", "Gamma", "Exponential", "Uniform", "Poisson", "Salt & Pepper"]
        for op in noise_ops:
            if st.button(f"Add {op} Noise", key=f"add_{op}"):
                apply_operation(getattr(noise, f"add_{op.lower().replace(' & ', '_and_')}_noise"))

    with st.sidebar.expander("Denoise Operations"):
        denoise_ops = ["Gaussian", "Rayleigh", "Gamma", "Exponential", "Uniform", "Poisson", "Salt & Pepper"]
        for op in denoise_ops:
            if st.button(f"Remove {op} Noise", key=f"rem_{op}"):
                apply_operation(getattr(denoise, f"remove_{op.lower().replace(' & ', '_and_')}_noise"))

    with st.sidebar.expander("Linear Filters"):
        linear_filters = ["Arithmetic", "Geometric", "Harmonic", "Contraharmonic"]
        for f in linear_filters:
            if st.button(f, key=f"lin_{f}"):
                apply_operation(getattr(filters, f"apply_{f.lower()}_filter"))

    with st.sidebar.expander("Non-Linear Filters"):
        nonlinear_filters = ["Median", "Min", "Max", "Midpoint", "Alpha Trimmed Mean", "Adaptive"]
        for f in nonlinear_filters:
            if st.button(f, key=f"nlin_{f}"):
                apply_operation(getattr(filters, f"apply_{f.replace(' ', '_').lower()}_filter"))

    with st.sidebar.expander("Frequency Filters"):
        freq_filters = ["Bandpass", "Bandreject", "Wiener", "Kalman"]
        for f in freq_filters:
            if st.button(f, key=f"freq_{f}"):
                apply_operation(getattr(filters, f"apply_{f.lower()}_filter"))


# --- Main Page Layout ---
if st.session_state.original_image is None:
    st.title("Welcome to Image Flare!")
    st.markdown("### Enhances and transforms your images with ease.")
    st.markdown("#### Let's upload an image from the sidebar to get started.")
else:
    # Image Display Area
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_container_width=True) # CORRECTED
    with col2:
        st.subheader("Processed Image")
        st.image(st.session_state.processed_image, use_container_width=True) # CORRECTED
    st.markdown("---")
    st.header("Control Panel")

    # --- Row 1: Main Actions ---
    c1, c2, c3, c4, c5 = st.columns(5)
    
    can_undo = st.session_state.history_index > 0
    can_redo = st.session_state.history_index < len(st.session_state.history) - 1
    if c1.button("Undo", disabled=not can_undo, use_container_width=True): # Button syntax is still correct
        st.session_state.history_index -= 1
        st.session_state.processed_image = st.session_state.history[st.session_state.history_index]
        st.rerun()
    if c2.button("Redo", disabled=not can_redo, use_container_width=True):
        st.session_state.history_index += 1
        st.session_state.processed_image = st.session_state.history[st.session_state.history_index]
        st.rerun()
    
    if c3.button("Reset", use_container_width=True):
        st.session_state.processed_image = st.session_state.original_image.copy()
        add_to_history(st.session_state.processed_image)
        st.rerun()

    buf = io.BytesIO()
    st.session_state.processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    c4.download_button(
        label="Save Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png",
        use_container_width=True
    )

    if c5.button("Convert to Grayscale", use_container_width=True):
        apply_operation(ImageOps.grayscale)

    # --- Row 2: Resize and Enhancements ---
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Resize Image")
        width = st.number_input("Width", value=st.session_state.processed_image.width, min_value=1)
        height = st.number_input("Height", value=st.session_state.processed_image.height, min_value=1)
        if st.button("Apply Resize", use_container_width=True):
            apply_operation(enhancements.resize_image, width=width, height=height)

    with c2:
        st.subheader("Enhancements")
        brightness = st.slider("Brightness", 0.1, 3.0, 1.0)
        contrast = st.slider("Contrast", 0.1, 3.0, 1.0)
        sharpness = st.slider("Sharpness", 0.0, 5.0, 1.0)
        blur_radius = st.slider("Blurriness", 0.0, 10.0, 0.0)
        if st.button("Apply Enhancements", use_container_width=True):
            temp_image = st.session_state.processed_image
            if brightness != 1.0: temp_image = enhancements.adjust_brightness(temp_image, brightness)
            if contrast != 1.0: temp_image = enhancements.adjust_contrast(temp_image, contrast)
            if sharpness != 1.0: temp_image = enhancements.adjust_sharpness(temp_image, sharpness)
            if blur_radius > 0.0: temp_image = enhancements.apply_blur(temp_image, blur_radius)
            st.session_state.processed_image = temp_image
            add_to_history(st.session_state.processed_image)
            st.rerun()
    
# --- Row 3: Analysis ---
st.markdown("---")
with st.expander("Analysis"):
    if st.button("Generate Histogram Comparison"):
        fig = analysis.create_histogram_comparison(st.session_state.original_image, st.session_state.processed_image)
        st.pyplot(fig)
    if st.button("Generate Quality Metrics"):
        fig = analysis.create_statistical_table(st.session_state.original_image, st.session_state.processed_image)
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Live Noise & Filter Comparison")
    
    if st.session_state.original_image is not None:
        st.info("This tool lets you test noise and filters without affecting your main processed image.")
        col1, col2, col3 = st.columns(3)

        # --- Define all available operations ---
        all_noise_ops = [f"Add {op} Noise" for op in ["Gaussian", "Rayleigh", "Gamma", "Exponential", "Uniform", "Poisson", "Salt & Pepper"]]
        all_denoise_ops = [f"Remove {op} Noise" for op in ["Gaussian", "Rayleigh", "Gamma", "Exponential", "Uniform", "Poisson", "Salt & Pepper"]]
        all_filter_ops = [
            "Arithmetic", "Geometric", "Harmonic", "Contraharmonic", "Median", "Min", "Max", 
            "Midpoint", "Alpha Trimmed Mean", "Adaptive", "Bandpass", "Bandreject", "Wiener", "Kalman"
        ]

        # Column 1: Original Image
        # Column 1: Original Image
        with col1:
             # Add vertical space to align with selectbox in col2 and col3
            st.markdown("<br><br><br>", unsafe_allow_html=True)  # Adjust number of <br> as needed
            st.image(st.session_state.original_image, caption="Original Image", use_container_width=True)
            fig_orig = analysis.create_single_histogram(st.session_state.original_image, "Original Histogram")
            st.pyplot(fig_orig)
        # Column 2: Noisy/Denoised Image (Generated on-the-fly)
        with col2:
            # UPDATED: Combined noise and denoise options into one list
            operation_type_live = st.selectbox(
                "Select Operation",
                options=["None"] + all_noise_ops + all_denoise_ops,
                key="live_operation_select"
            )
            
            image_col2 = st.session_state.original_image
            caption_text_col2 = "No Operation Applied"

            if operation_type_live != "None":
                caption_text_col2 = operation_type_live

                # Logic to decide whether to add noise or denoise
                if operation_type_live.startswith("Add"):
                    op_name = operation_type_live.replace("Add ", "").replace(" Noise", "")
                    func_name = f"add_{op_name.lower().replace(' & ', '_and_')}_noise"
                    op_func = getattr(noise, func_name)
                    image_col2 = op_func(st.session_state.original_image)
                elif operation_type_live.startswith("Remove"):
                    op_name = operation_type_live.replace("Remove ", "").replace(" Noise", "")
                    func_name = f"remove_{op_name.lower().replace(' & ', '_and_')}_noise"
                    op_func = getattr(denoise, func_name)
                    image_col2 = op_func(st.session_state.original_image)

            st.image(image_col2, caption=caption_text_col2, use_container_width=True)
            fig_noisy = analysis.create_single_histogram(image_col2, "Column 2 Histogram")
            st.pyplot(fig_noisy)
            
        # Column 3: Filtered Image (Generated on-the-fly) - This column is unchanged as requested
        with col3:
            filter_type_live = st.selectbox(
                "Select Filter to Apply",
                options=["None"] + all_filter_ops,
                key="live_filter_select"
            )

            filtered_image_live = image_col2 # Start with the image from column 2
            caption_text_col3 = "No Filter Applied"

            if filter_type_live != "None":
                caption_text_col3 = f"{filter_type_live} Filter Applied"
                func_name = f"apply_{filter_type_live.replace(' ', '_').lower()}_filter"
                filter_func = getattr(filters, func_name)
                filtered_image_live = filter_func(image_col2)

            st.image(filtered_image_live, caption=caption_text_col3, use_container_width=True)
            fig_filtered = analysis.create_single_histogram(filtered_image_live, "Filtered Image Histogram")
            st.pyplot(fig_filtered)