# MNIST-based digit recognizer with drawing canvas and DB logging
Try me out [here](http://nikolaskuhn.com/digit-app).

This is a simple web app that lets the user draw a digit and then tries to recognize the digit using a simple neural net. The user can also provide the correct answer, in which case the result of the prediction gets stored in an internal database.

# Running Locally with Docker (Recommended)

The easiest way to run the application locally is by using Docker Compose. One can also run the app directly using python and streamlit, but this requires more setting up a dedicated database.

## Setup (using Docker)
### 0. Prerequisites
*   Docker Desktop (or Docker Engine + Docker Compose CLI) installed and running

### 1. Clone Repository
* Clone the repository in your desired location:
```bash
git clone https://github.com/nick-kuhn/digit-app
cd digit-app
```

### 2. Configure Environment Variables
Create and configure the `.env` file for Docker Compose:

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Change the value `example_password` to your desired password.
4.  The example values  `default_user` and `mnist_logs` can also be adjusted if desired. 

## Running the application (using Docker)
### 1. Build and run the application containers
From the project directory (where `docker-compose.yml` is located):
```bash
docker-compose up --build -d
```
This also sets up a PostgreSQL volume managed by Docker, which persists between runs
### 2. Access the application
Once the containers are up and running (this might take a bit the first time when the image is built), the application will be accessible in your web browser at:
[http://localhost:8501](http://localhost:8501)


### 3. Stopping the application
To stop and remove the containers:
```bash
docker-compose down
```
If you also want to remove the named volume where PostgreSQL data is stored (e.g. to start completely fresh), use:
```bash
docker-compose down -v
```


# Notes on the model (mnist_cnn.pth).
The model is a Convolutional Neural Network (CNN) trained on the MNIST digit database. It uses two convolutional layers with, followed by max pooling. Dropout layers are used for regularization before the features are flattened and passed through two fully connected layers. The final layer outputs log probabilities for each of the 10 digit classes (0-9), indicating the model's prediction.

The accuracy is ca. 93 % on MNIST data, but is likely lower for digits drawn in the app.