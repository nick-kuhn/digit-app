import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms # Added for Normalize transform if not implicitly in test_data

# Assuming model.py and mnist.py are in the same directory or accessible in PYTHONPATH
from model import Net
from mnist import test_data

def get_model_accuracy(model_path='mnist_cnn.pth', batch_size=64):
    """
    Loads a trained model, evaluates it on the MNIST test dataset, and prints the accuracy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # Instantiate the model
    model = Net().to(device)

    # Load the trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please ensure the model path is correct and the file exists.")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    model.eval() # Set the model to evaluation mode

    # Prepare the test dataloader
    # The test_data from mnist.py already has ToTensor transform.
    # If normalization was used during training, it should be applied here too.
    # The streamlit_app.py uses transforms.Normalize((0.1307,), (0.3081,))
    # We should be consistent.
    # Let's check how test_data is defined in mnist.py before adding a transform here.
    # For now, assuming test_data is ready or only needs batching.
    
    # Re-checking mnist.py content:
    # test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    # It seems normalization is not applied directly in mnist.py's test_data.
    # However, training.py uses test_dataloader from prepare_data.py, which uses test_data from mnist.py.
    # And streamlit_app.py which loads mnist_cnn.pth does apply normalization.
    # For accurate evaluation, the same preprocessing (including normalization) used
    # during training and for the saved model should be used here.

    # Let's apply the same normalization as in streamlit_app.py for consistency
    # if it's not already part of test_data's transforms.
    # Since test_data from mnist.py only has ToTensor(), we add Normalize.
    
    # We need to ensure test_data has the right transforms.
    # The `test_data` from `mnist.py` is already a `Dataset` object.
    # We can create a DataLoader from it directly.
    # The normalization transform should be applied if it was used during the training of mnist_cnn.pth
    # The `streamlit_app.py` uses:
    # transform = transforms.Compose([
    #     transforms.ToTensor(), # This is already in mnist.py's test_data
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # So, we should ensure test_data uses this normalization.
    # A clean way is to get the raw test_data and apply the full expected transform.

    # Re-importing datasets to apply consistent transform
    from torchvision import datasets
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and std
    ])
    
    # Using fresh dataset object with proper transform to be sure
    mnist_test_dataset = datasets.MNIST(
        root="data", # Assuming 'data' directory for MNIST
        train=False,
        download=True, # Will use cached if already downloaded
        transform=transform
    )

    test_dataloader = DataLoader(mnist_test_dataset, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss() # Or nn.NLLLoss() if model output is log_softmax
    # The model.py uses F.log_softmax, so NLLLoss is appropriate.
    # The training.py uses nn.CrossEntropyLoss which combines log_softmax and NLLLoss.
    # Since our model's forward pass ends with log_softmax, NLLLoss is the correct choice here.
    # Let's verify model.py's output: output = F.log_softmax(x, dim=1) - Yes, it is log_softmax.
    loss_fn = nn.NLLLoss()


    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # Ensure no gradients are computed during evaluation
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct

    print(f"Test Results: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f}")
    return accuracy, test_loss

if __name__ == '__main__':
    get_model_accuracy()
