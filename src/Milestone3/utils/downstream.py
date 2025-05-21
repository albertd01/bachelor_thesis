import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, roc_auc_score
import torch

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128, output_dim=1):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

def run_downstream_task(ecfp_array, ngf_array, labels, task_type, hidden_dim=128, k=10):
    scores_ecfp = []
    scores_ngf = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(labels):
        X_train_ecfp, X_test_ecfp = ecfp_array[train_idx], ecfp_array[val_idx]
        X_train_ngf, X_test_ngf = ngf_array[train_idx], ngf_array[val_idx]
        y_train, y_test = labels[train_idx], labels[val_idx]

        if task_type == 'regression':
            ecfp_model = MLPRegressor(input_dim=ecfp_array.shape[1], hidden_dim=hidden_dim)
            ngf_model = MLPRegressor(input_dim=ngf_array.shape[1], hidden_dim=hidden_dim)
        else:
            ecfp_model = MLPClassifier(input_dim=ecfp_array.shape[1], hidden_dim=hidden_dim)
            ngf_model = MLPClassifier(input_dim=ngf_array.shape[1], hidden_dim=hidden_dim)

        score_ecfp = train_model(ecfp_model, X_train_ecfp, y_train, X_test_ecfp, y_test, task_type)
        score_ngf = train_model(ngf_model, X_train_ngf, y_train, X_test_ngf, y_test, task_type)

        scores_ecfp.append(score_ecfp)
        scores_ngf.append(score_ngf)

    return {
        "ecfp": (np.mean(scores_ecfp), np.std(scores_ecfp)),
        "ngf": (np.mean(scores_ngf), np.std(scores_ngf))
    }


def train_model(model, X_train, y_train, X_test, y_test, task_type, epochs=100, lr=1e-3):
    model = model.to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32 if task_type == 'regression' else torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32 if task_type == 'regression' else torch.long)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        if task_type == 'regression':
            loss = F.mse_loss(out, y_train)
        else:
            out = out.view(-1)
            loss = F.binary_cross_entropy_with_logits(out, y_train.float())
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        if task_type == 'classification':
            preds = torch.sigmoid(preds).view(-1).numpy()
            score = roc_auc_score(y_test.numpy(), preds)
        else:
            preds = preds.numpy()
            score = root_mean_squared_error(y_test.numpy(), preds) 
    return score
