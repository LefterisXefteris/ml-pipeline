# ml-pipeline

## 🔑 Key Takeaways:

- Built a **Convolutional Neural Network (CNN)** to train a computer vision model that recognizes digits drawn by the user.
  
- Used **PyTorch's `transforms v2`** to preprocess the dataset — including converting images to tensors, scaling pixel values, and normalizing them — to improve training performance.

- Trained the model on the **MNIST dataset** of handwritten digits, using the training set and validating it against the test set.

- After **15 epochs**, the model reached approximately **95% accuracy**.

- Created a **Streamlit web app** with a drawable canvas that allows users to sketch a digit and get a prediction from the trained model in real-time.  
  The use of proper transforms during inference significantly improved prediction quality.

- **Next step**: I’m working on deploying the model using **FastAPI** for the backend and integrating a **PostgreSQL** database to log user predictions and track analytics.  
  I'm also exploring ways to further improve the model's performance and generalization.