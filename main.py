import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import glob

pd.set_option('future.no_silent_downcasting', True)


def load_data(files):
    # Load and combine CSVs
    #files = glob.glob("./data/*.csv")
    #files = ["./data/loan_approval_0.csv", "./data/loan_approval_1.csv"]
    df = pd.concat([pd.read_csv(f, sep="|") for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()

    # Drop identifier column (not a feature)
    df.drop(columns=['Loan_ID'], inplace=True)

    # Map categorical variables
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({True: 1, False: 0})
    df['Self_Employed'] = df['Self_Employed'].map({True: 1, False: 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})

    df = pd.get_dummies(df, columns=['Property_District'])
    # df['Property_District'] = pd.factorize(df['Property_District'])[0]

    # Replace common placeholders with 0
    df = df.replace(['-', '', 'NA', 'NaN', 'nan'], 0).infer_objects(copy=False)

    # Convert all columns to numeric (invalids become NaN)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill any remaining NaNs
    df.fillna(0, inplace=True)

    # Drop columns that are entirely constant
    df = df.loc[:, df.nunique() > 1]

    print("Columns used:", df.columns.tolist())
    print("Shape:", df.shape)

    # Check class balance
    target = df.iloc[:, -1]
    n_pos = (target == 1).sum()
    n_neg = (target == 0).sum()
    print(f"Class balance — 1s: {n_pos}, 0s: {n_neg}, ratio: {n_pos/n_neg:.2f}")

    # Normalize features (0-1) with epsilon to avoid 0/0
    features = df.iloc[:, :-1]
    features = features.astype(float)
    col_min = features.min()
    col_max = features.max()
    features = (features - col_min) / (col_max - col_min + 1e-8)

    X_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(target.values, dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor, n_pos, n_neg

def split_train_test(X_tensor, y_tensor):

    n = len(X_tensor)
    split = int(n * 0.8)
    idx = torch.randperm(n)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_test,  y_test  = X_tensor[test_idx],  y_tensor[test_idx]
    return (X_train, y_train), (X_test, y_test)



def test(model, X_train, X_test, y_train, y_test):

    model.eval()
    with torch.no_grad():
        train_preds = (torch.sigmoid(model(X_train)) > 0.5).float()
        test_preds  = (torch.sigmoid(model(X_test))  > 0.5).float()

        train_acc = (train_preds == y_train).float().mean()
        test_acc  = (test_preds  == y_test ).float().mean()

        tp = ((test_preds == 1) & (y_test == 1)).sum().item()
        tn = ((test_preds == 0) & (y_test == 0)).sum().item()
        fp = ((test_preds == 1) & (y_test == 0)).sum().item()
        fn = ((test_preds == 0) & (y_test == 1)).sum().item()

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f}")
    print(f"\nConfusion Matrix (test):")
    print(f"               Predicted 1 | Predicted 0")
    print(f"  Actual 1:    {tp:5d}      |  {fn:5d}")
    print(f"  Actual 0:    {fp:5d}      |  {tn:5d}")

def test_new(model, X_new, y_new):
    model.eval()
    with torch.no_grad():
        train_preds = (torch.sigmoid(model(X_new)) > 0.5).float()
        train_acc = (train_preds == y_new).float().mean()
    print(train_acc)



def train(model, n_neg, n_pos, X_train, y_train, X_test, y_test, epochs, loss_break, lr):

    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < loss_break:
            print(f"Early stop at epoch {epoch+1} — loss: {loss.item():.4f}")
            break

        if (epoch + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test), y_test).item()
            print(f"Epoch {epoch+1:4d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f}")


class NN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, output_size),
        )

    def forward(self, x):
        return self.net(x)



def main():

    X_tensor, y_tensor, n_pos, n_neg = load_data(glob.glob("./data/*.csv"))
    (X_train, y_train), (X_test, y_test) = split_train_test(X_tensor, y_tensor)

    epochs     = 3_000
    input_size = X_tensor.shape[1]
    hidden1    = input_size * 2
    hidden2    = input_size * 2
    output_size = 1
    lr         = 1e-3
    loss_break  = 0.01

    model = NN(input_size, hidden1, hidden2, output_size)
    train(model, n_neg, n_pos, X_train, y_train, X_test, y_test, epochs, loss_break, lr)
    test(model, X_train, X_test, y_train, y_test)

    X_new, y_new, _, _ = load_data(glob.glob("./data/*.csv"))
    test_new(model, X_new, y_new)


if __name__ == "__main__":
    main()

