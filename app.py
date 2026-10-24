
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="XraySyntheticAI",
    page_icon="🩻",
    layout="wide"
)

st.title("🩻 XraySyntheticAI")
st.subheader("Synthetic X-ray Generation for Rare Classes")

st.write(
    "An intelligent system to analyze class imbalance "
    "and prepare synthetic X-ray image generation."
)

DATA_DIR = Path("data/chest_xray")

if DATA_DIR.exists():
    st.success("Dataset folder detected!")

    for split in ["train", "val", "test"]:
        split_path = DATA_DIR / split

        if split_path.exists():
            st.write(f"### {split.title()} Dataset")

            counts = {}

            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    counts[class_dir.name] = sum(
                        1 for p in class_dir.iterdir()
                        if p.is_file()
                        and p.suffix.lower()
                        in [".jpg", ".jpeg", ".png"]
                    )

            if counts:
                df = pd.DataFrame(
                    list(counts.items()),
                    columns=["Class", "Image Count"]
                )

                st.dataframe(df, use_container_width=True)
                st.bar_chart(
                    df.set_index("Class")["Image Count"]
                )
else:
    st.warning(
        "Dataset not found yet. Add the extracted dataset "
        "to data/chest_xray."
    )

st.divider()

st.header("Synthetic Image Generation")

target_class = st.selectbox(
    "Select the target class",
    ["PNEUMONIA", "NORMAL"]
)

target_count = st.number_input(
    "Desired total number of images",
    min_value=1,
    value=100,
    step=10
)

st.info(
    f"Selected class: {target_class}. "
    f"Target count: {target_count}. "
    "The AI generator will be connected in the next stage."
)
