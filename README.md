# ml-pipeline

## 🔑 Key Takeaways:

- Built a **Convolutional Neural Network (CNN)** to train a computer vision model that recognizes digits drawn by the user.
  
- Used **PyTorch's `transforms v2`** to preprocess the dataset — including converting to tensors, scaling pixel values, and normalizing them — to improve training performance.

- Trained the model on the **MNIST dataset** for handwritten digits, using the training set and validating it against the test set.

- After **15 epochs**, the model reached approximately **95% accuracy**.

- Created a **Streamlit web app** with a drawable canvas that allows users to sketch a digit and get a prediction from the trained model in real-time.

- **Next step**: I’m working on deploying the model using **FastAPI** for the backend and connecting it to a **PostgreSQL** database to log user predictions and manage analytics.  
Still figuring out the best way to structure this, and figuring out how can imporve the model.