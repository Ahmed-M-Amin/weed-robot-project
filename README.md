# Weed Robot Project 🤖🌱

AI-powered detection of weeds and plants for smart farming, with a modern dashboard, easy retraining, and clear results.

---

## 📁 Project Structure

```
weed-robot-project/
│
├── .pytest_cache/           # Pytest cache
├── __pycache__/             # Python cache
├── check/                   # Test images for detection
├── data/                    # Training and validation images (not included)
├── simulation_output/       # Output images from detection
├── venv/                    # Python virtual environment (should be ignored)
│
├── dashboard.py             # Streamlit dashboard GUI
├── main.py                  # Main project file
├── requirements.txt         # Required Python packages
├── robot.jpg                # Project logo (shows in GUI)
├── test_project.py          # Unit/integration tests
├── train_model.py           # Model training script
├── weed_plant_detector.keras# Trained model (see download instructions)
```

---

## 🚀 Getting Started

1. **Clone this repository**

   ```bash
   git clone https://github.com/Ahmed-M-Amin/weed-robot-project.git
   cd weed-robot-project
   ```

2. **(Optional) Create & activate a virtual environment**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**

   - Place your training/validation images in the `data/` directory (not included in this repo).
   - The data folder should follow the structure expected by the training script (`train_model.py`).

5. **Download the trained model**

   - Go to the [latest GitHub Release](https://github.com/Ahmed-M-Amin/weed-robot-project/releases/latest) and download `weed_plant_detector.keras`.
   - Place the file in your project root directory.

6. **Run the Dashboard**

   ```bash
   streamlit run dashboard.py
   ```

   - Open the local URL shown in your terminal.

7. **Train a new model (optional)**

   ```bash
   python train_model.py
   ```

   - This will save a new `weed_plant_detector.keras` model file.

8. **Run tests**

   ```bash
   python test_project.py
   ```

---

## 🖼️ Features

- **Weed & Plant Detection:**

  - Green boxes for detected plants, red for weeds (see `simulation_output/` for examples).

- **Interactive Dashboard:**

  - Upload images, see instant results and detection statistics.

- **Customizable:**

  - Retrain the model on your own data using `train_model.py`.

- **Professional UI:**

  - Robot logo and clean, modern layout.

---

## 💡 Model File (`.keras`)

- Download the model from the [latest GitHub Release](https://github.com/Ahmed-M-Amin/weed-robot-project/releases/latest).
- The dashboard and main program will use `weed_plant_detector.keras` for predictions.

---

## 📝 .gitignore Recommendation

```
venv/
__pycache__/
*.pyc
data/
simulation_output/
```

---

## 👥 Authors

- **Ahmed Abdrabou**

---

## 📞 Contact

For questions, open an [issue](https://github.com/Ahmed-M-Amin/weed-robot-project/issues) or contact the maintainers.
