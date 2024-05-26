


# Paths for saving data and models
DATASET_PATH = "home/huilin/project/data"
AE_MODEL_PATH = CHECKPOINT_PATH = "home/huilin/project/AE-model"
# Added to save the model



# Loading the training dataset. We need to split it into a training and validation part
train_dataset = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [55000, 5000])

# Loading the test set
test_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

def train_mnist(latent_dim):
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"mnist_{latent_dim}.ckpt")
    
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = Autoencoder.load_from_checkpoint(pretrained_filename)