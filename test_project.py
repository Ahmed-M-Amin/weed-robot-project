import os
import shutil
import tempfile
import numpy as np
import pytest
import tensorflow as tf

# --- Import your project code ---
import train_model
import main

# --- Unit Test: Model Creation ---
def test_create_model():
    """Test the creation of the Keras model."""
    model = train_model.create_model((128, 128, 3), 2)
    assert isinstance(model, tf.keras.Model)
    # Model should have output shape (None, 2)
    assert model.output_shape[-1] == 2

# --- Integration Test: Dummy Data Directory Creation ---
def test_create_dummy_dataset():
    """Test that the dummy dataset structure is created."""
    tmp_dir = tempfile.mkdtemp()
    try:
        train_model.create_dummy_dataset(tmp_dir)
        # Check that folders exist
        assert os.path.isdir(os.path.join(tmp_dir, 'train/weeds'))
        assert os.path.isdir(os.path.join(tmp_dir, 'train/plants'))
        assert os.path.isdir(os.path.join(tmp_dir, 'validation/weeds'))
        assert os.path.isdir(os.path.join(tmp_dir, 'validation/plants'))
    finally:
        shutil.rmtree(tmp_dir)

# --- Unit Test: Model Saving and Loading ---
def test_train_and_load_model():
    """Test training (on fake data) and reloading model."""
    # Create a temp data directory with one small random image per class
    tmp_data = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_data, "train/weeds"), exist_ok=True)
    os.makedirs(os.path.join(tmp_data, "train/plants"), exist_ok=True)
    os.makedirs(os.path.join(tmp_data, "validation/weeds"), exist_ok=True)
    os.makedirs(os.path.join(tmp_data, "validation/plants"), exist_ok=True)

    fake_img = np.ones((128, 128, 3), dtype=np.uint8) * 255
    tf.keras.preprocessing.image.save_img(os.path.join(tmp_data, "train/weeds/weed.jpg"), fake_img)
    tf.keras.preprocessing.image.save_img(os.path.join(tmp_data, "train/plants/plant.jpg"), fake_img)
    tf.keras.preprocessing.image.save_img(os.path.join(tmp_data, "validation/weeds/weed.jpg"), fake_img)
    tf.keras.preprocessing.image.save_img(os.path.join(tmp_data, "validation/plants/plant.jpg"), fake_img)

    tmp_model = os.path.join(tmp_data, "test_model.keras")
    try:
        # Should train and save without error
        train_model.train_model(data_dir=tmp_data, model_save_path=tmp_model)
        # Should be able to reload the model
        model = tf.keras.models.load_model(tmp_model)
        assert isinstance(model, tf.keras.Model)
    finally:
        shutil.rmtree(tmp_data)

# --- System Test: End-to-End Simulation with Main ---
def test_run_simulation_with_fake_image():
    """Test the main run_simulation function with a fake image and model."""
    tmp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(tmp_dir, "output")
    fake_img_path = os.path.join(tmp_dir, "fake.jpg")

    # Create a white fake image
    fake_img = np.ones((640, 480, 3), dtype=np.uint8) * 255
    import cv2
    cv2.imwrite(fake_img_path, fake_img)

    # Temporarily patch main.load_detection_model to return a dummy model
    orig_load_model = main.load_detection_model
    main.load_detection_model = lambda: None  # Forces fallback to color detection

    try:
        # Should process the image and produce output files
        main.run_simulation([fake_img_path], output_dir=output_dir)
        assert os.path.exists(os.path.join(output_dir, "processed_frame_0.jpg"))
        assert os.path.exists(os.path.join(output_dir, "robot_log.txt"))
    finally:
        main.load_detection_model = orig_load_model
        shutil.rmtree(tmp_dir)

# --- Optional: Add more tests for individual logic (robot_navigate, predict_objects) as needed ---
