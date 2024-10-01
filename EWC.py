import torch
from torch.utils.data import DataLoader


class EWC:
    def __init__(self, model, dataset, device='cpu'):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self.compute_fisher_information()
        self.optimal_params = {n: p.clone().detach() for n, p in self.params.items()}

    def compute_fisher_information(self):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
        self.model.eval()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            for n, p in self.params.items():
                fisher[n] += p.grad.pow(2).detach()

        fisher = {n: f / len(self.dataset) for n, f in fisher.items()}  # Normalize fisher information
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += torch.sum(self.fisher[n] * (p - self.optimal_params[n]).pow(2))
        return loss
