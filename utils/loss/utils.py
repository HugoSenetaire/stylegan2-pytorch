import torch

def create_label(batch_size,column_size,device):
    # TODO
    # Non nécessaire de le créer à chaque fois , juste mettre en global
    labels = torch.tensor([i%column_size for i in range(batch_size*column_size)]).to(device)
    return labels
