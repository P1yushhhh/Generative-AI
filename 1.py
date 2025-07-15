import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))  # This should print your token!

import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)