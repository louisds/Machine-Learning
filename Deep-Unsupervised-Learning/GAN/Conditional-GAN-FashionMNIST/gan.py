#import necessary modules
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim as optim
# for visualization
from matplotlib import pyplot as plt
import math
import numpy as np

# tells PyTorch to use an NVIDIA GPU, if one is available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading the dataset
training_parameters = {
    "img_size": 28,
    "n_epochs": 50,
    "batch_size": 64,
    "learning_rate_generator": 0.0001,
    "learning_rate_discriminator": 0.0001,
}
# define a transform to 1) scale the images and 2) convert them into tensors
transform = transforms.Compose([
    # scales the smaller edge of the image to have this size
    transforms.Resize(training_parameters['img_size']), 
    transforms.ToTensor(),
])

# load the dataset
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        # specifies the directory to download the datafiles to, 
        # relative to the location of the notebook.
        './data', 
        train = True,
        download = True,
        transform = transform),
    batch_size = training_parameters["batch_size"],
    shuffle=True
    )

# Fashion MNIST has 10 classes, just like MNIST. Here's what they correspond to:
label_descriptions = {
      0: 'T-shirt/top',
      1	: 'Trouser',
      2	: 'Pullover',
      3	: 'Dress',
      4	: 'Coat',
      5	: 'Sandal',
      6	: 'Shirt',
      7	: 'Sneaker',
      8	: 'Bag',
      9	: 'Ankle boot'
}

# Create the Generator model class, which will be used to initialize generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_labels=10): 
        super(Generator,self).__init__() # initialize the parent class
        self.label_embedding = nn.Embedding(10, 10) 
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim + num_labels, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    def forward(self, x, labels=None):
        addlabels = self.label_embedding(labels)
        x = torch.cat([x,addlabels], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output.to(device)

class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_labels=10):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(10, 10)
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(input_dim + num_labels, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, labels=None): 
        addlabels = self.label_embedding(labels)
        x = torch.cat([x,addlabels], 1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output.to(device)

# Initialize both models, and load them to the GPU or CPU.
discriminator = Discriminator(784,1).to(device)
generator = Generator(100,784).to(device)

discriminator_optimizer = optim.Adam(discriminator.parameters(), 
                          lr=training_parameters['learning_rate_discriminator'])
generator_optimizer = optim.Adam(generator.parameters(), 
                              lr=training_parameters['learning_rate_generator'])

lossf = nn.BCELoss()
def train_generator(batch_size):
    """
    Performs a training step on the generator by
        1. Generating fake images from random noise.
        2. Running the discriminator on the fake images.
        3. Computing loss on the result.
    :arg batch_size: the number of training examples in the current batch
    Returns the average generator loss over the batch.
    """
    # Start by zeroing the gradients of the optimizer
    generator_optimizer.zero_grad()
    # Create a new batch of fake images 
    # (since the discriminator has just been trained on the old ones)
    noise = torch.randn(batch_size,100).to(device) 
    fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
    fake_data = generator.forward(noise, fake_labels).to(device)
    #fake_data = generator.forward(noise).to(device) 
    fake_data = fake_data.view(-1, 784)


    discriminator_on_fake = discriminator.forward(fake_data, 
                                                  fake_labels).view(batch_size)
    g_loss = lossf(discriminator_on_fake, torch.ones(batch_size).to(device))
    g_loss.backward()
    generator_optimizer.step()

    return g_loss

def train_discriminator(batch_size, images, labels=None): 
    """
    Performs a training step on the discriminator by
        1. Generating fake images from random noise.
        2. Running the discriminator on the fake images.
        3. Running the discriminator on the real images
        3. Computing loss on the results.
    :arg batch_size: the number of training examples in the current batch
    :arg images: the current batch of images, a tensor of size BATCH x 1 x 64x64
    :arg labels: the labels corresponding to images, a tensor of size BATCH
    Returns the average loss over the batch.
    """
    discriminator_optimizer.zero_grad()
    noise = torch.randn(batch_size,100).to(device)
    fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
    fake_data = generator(noise, fake_labels)
    #fake_data = generator.forward(noise).to(device) 
    fake_data = fake_data.view(-1, 784)
    real_data = images.view(batch_size, 784).to(device) 
    real_labels = labels.to(device)


    discriminator_on_fake = discriminator.forward(fake_data.detach(), 
                                                  fake_labels).view(batch_size)
    discriminator_on_real = discriminator.forward(real_data, 
                                                  real_labels).view(batch_size)

    fake_discriminator_loss = lossf(discriminator_on_fake, 
                                    torch.zeros(batch_size).to(device))
    real_discriminator_loss = lossf(discriminator_on_real, 
                                    torch.ones(batch_size).to(device))


    d_loss = (fake_discriminator_loss + real_discriminator_loss) / 2

    d_loss.backward()
    discriminator_optimizer.step()

    return d_loss


for epoch in range(training_parameters['n_epochs']):
    G_loss = []  # for plotting the losses over time
    D_loss = []
    for batch, (imgs, labels) in enumerate(train_loader):
        batch_size = labels.shape[0] 
        lossG = train_generator(batch_size)
        G_loss.append(lossG)
        lossD = train_discriminator(batch_size, imgs, labels)
        D_loss.append(lossD)

        if ((batch + 1) % 500 == 0 and (epoch + 1) % 2 == 0):
            # Display a batch of generated images and print the loss
            print("Training Steps Completed: ", batch)
            # Disables gradient computation to speed things up
            with torch.no_grad():  
                noise = torch.randn(batch_size, 100).to(device)
                fake_labels = torch.randint(0, 10, (batch_size,)).to(device)
                generated_data = generator(noise, 
                                     fake_labels).cpu().view(batch_size, 28, 28)

                # display generated images
                batch_sqrt = int(training_parameters['batch_size'] ** 0.5)
                fig, ax = plt.subplots(batch_sqrt, batch_sqrt, figsize=(15, 15))
                for i, x in enumerate(generated_data):
                    ax[math.floor(i / batch_sqrt)][i % batch_sqrt].set_title(label_descriptions[int(fake_labels[i].item())]) 
                    ax[math.floor(i / batch_sqrt)][i % batch_sqrt].imshow(x.detach().numpy(), 
                                                                          interpolation='nearest', cmap='gray')
                    ax[math.floor(i / batch_sqrt)][i % batch_sqrt].get_xaxis().set_visible(False)
                    ax[math.floor(i / batch_sqrt)][i % batch_sqrt].get_yaxis().set_visible(False)
                plt.show()
                #fig.savefig(f"./results/CGAN_Generations_Epoch_{epoch}")
                print(f"Epoch {epoch}: loss_d: {torch.mean(torch.FloatTensor(D_loss))}, loss_g: {torch.mean(torch.FloatTensor(G_loss))}")