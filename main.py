import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import random
import pandas as pd
import numpy as np
import os
import datetime
from torch_scatter import scatter
from pytorchtools import EarlyStopping
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from mlDataset import MLDataset

# CONFIG
# top k items to recommend
TOP_K = 10

DATASET_PATH = "./movielens100k.csv"

# Model parameters
PRETRAIN_EPOCHS = 50
POISON_EPOCHS = 30
BATCH_SIZE = 4096

SEED = 42

# Poisoning Parameters
kappa = 0.9
eta = 3
delta = 1.0
TARGET = 848

# number of fake users
m = 10


# NemMF params
n_factors = 64
hidden_size = 128

num_negative_items = 4

pretrain_lr = 0.001
poison_training_lr = 0.001

n_target_items = 30

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


class NeuralNetwork(nn.Module):
    def __init__(self, num_factors, num_users, num_items):
        super(NeuralNetwork, self).__init__()
        self.P = nn.Embedding(num_users + 1, num_factors)
        self.Q = nn.Embedding(num_items + 1, num_factors)
        self.U = nn.Embedding(num_users + 1, num_factors)
        self.V = nn.Embedding(num_items + 1, num_factors)
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(num_factors * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(hidden_size, num_factors),
            nn.ReLU(),
        )
        self.prediction_layer = nn.Sequential(
            nn.Linear(num_factors * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat([p_mlp, q_mlp], dim=1))
        con_res = torch.cat([gmf, mlp], dim=1)

        return self.prediction_layer(con_res)


def make_splits(df, train_ratio=0.75, min_n_items=10):
    train_userID = []
    train_itemID = []
    train_rating = []
    train_timestamp = []
    test_userID = []
    test_itemID = []
    test_rating = []
    test_timestamp = []

    train_ratio = 0.75
    split_num = 1 / (1 - train_ratio)

    for i in range(1, 943 + 1):
        j_count = 0
        for j in df[df["userID"] == i].iterrows():
            if j_count < min_n_items:
                # train append
                train_userID.append(j[1][0])
                train_itemID.append(j[1][1])
                train_rating.append(j[1][2])
                train_timestamp.append(j[1][3])

            elif j_count < min_n_items * 2:
                # test append
                test_userID.append(j[1][0])
                test_itemID.append(j[1][1])
                test_rating.append(j[1][2])
                test_timestamp.append(j[1][3])
            else:
                if (j_count + 1) % split_num == 0:
                    # test append
                    test_userID.append(j[1][0])
                    test_itemID.append(j[1][1])
                    test_rating.append(j[1][2])
                    test_timestamp.append(j[1][3])
                else:
                    # train append
                    train_userID.append(j[1][0])
                    train_itemID.append(j[1][1])
                    train_rating.append(j[1][2])
                    train_timestamp.append(j[1][3])

            j_count += 1
    train_df = pd.DataFrame(
        data={
            "userID": train_userID,
            "itemID": train_itemID,
            "rating": train_rating,
            "timestamp": train_timestamp,
        }
    )
    test_df = pd.DataFrame(
        data={
            "userID": test_userID,
            "itemID": test_itemID,
            "rating": test_rating,
            "timestamp": test_timestamp,
        }
    )
    return train_df, test_df


class HeuristicHitRatio(nn.Module):
    def __init__(self):
        super(HeuristicHitRatio, self).__init__()

    def forward(self, user_id, item_id, rating, preds, y_u_t, y_v):

        y_v_22 = torch.pow(torch.linalg.vector_norm(y_v), 2)

        min_log_y_hat_ui = torch.log(scatter(preds, user_id, reduce="min"))

        log_y_u_t = torch.log(scatter(y_u_t, user_id, reduce="mean"))

        l_u_max_left = torch.sub(min_log_y_hat_ui, log_y_u_t)

        l_u_max_right = torch.full_like(l_u_max_left, -1.0)

        l_u = torch.max(l_u_max_left, l_u_max_right)

        l_dash = nn.BCELoss()

        l_dash_value = l_dash(preds, rating.detach())

        print(torch.sum(l_u), l_dash_value, y_v_22)

        return torch.sum(l_u) + 1000 * l_dash_value + y_v_22


def f_test(
    data_loader, model, loss_fn, selected_items=[], selected_userIDs=[], pretrain=False
):
    all_preds = []
    target_preds = []
    target_item = torch.full((BATCH_SIZE,), fill_value=TARGET).to(device)
    n_records = len(data_loader) * BATCH_SIZE
    y_v = []
    y_v_toLoss = torch.tensor([0.0]).clone().to(device)
    additionalZeros = 0
    model.eval()
    with torch.no_grad():
        for user, item, rating in data_loader:
            X_user = torch.from_numpy(np.array(user)).clone().to(device)
            X_item = torch.from_numpy(np.array(item)).clone().to(device)

            if len(X_user) < BATCH_SIZE:
                additionalZeros = (
                    torch.full((BATCH_SIZE - len(X_user),), fill_value=0)
                    .clone()
                    .to(device)
                )
                additionalZerosAlt = (
                    torch.full((BATCH_SIZE - len(X_user), 1), fill_value=0)
                    .clone()
                    .to(device)
                )
                target_itemAlt = torch.full(
                    (BATCH_SIZE - (BATCH_SIZE - len(X_user)),), fill_value=TARGET
                ).to(device)
                pred = model(X_user, X_item).to(device)
                pred = torch.cat([pred, additionalZerosAlt], dim=0).clone().to(device)
                all_preds.append(pred)
                t_predictions = model(X_user, target_itemAlt).clone().to(device)
                t_predictions = (
                    torch.cat([t_predictions, additionalZerosAlt], dim=0)
                    .clone()
                    .to(device)
                )
                target_preds.append(t_predictions)
            else:
                pred = model(X_user, X_item).to(device)
                all_preds.append(pred)
                target_preds.append(model(X_user, target_item).clone().to(device))
        if n_users > 944:
            X_user_v = torch.from_numpy(np.array(selected_userIDs)).clone().to(device)
            X_item_v = torch.from_numpy(np.array(selected_items)).clone().to(device)
            y_v_toLoss = model(X_user_v, X_item_v).clone().to(device)

    predsTensor = torch.reshape(torch.stack(all_preds, dim=-1), (n_records,)).to(device)
    targetItemTensor = torch.reshape(
        torch.stack(target_preds, dim=-1), (n_records,)
    ).to(device)
    if pretrain == True:
        if type(additionalZeros) == torch.Tensor:
            loss = loss_fn(
                predsTensor,
                torch.cat(
                    [data_loader.dataset.rating.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
                .detach(),
            )
        else:

            loss = loss_fn(
                predsTensor, data_loader.dataset.rating.clone().to(device).detach()
            )
    else:
        if type(additionalZeros) == torch.Tensor:
            userID = (
                torch.cat(
                    [data_loader.dataset.user_id.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
            )
            itemID = (
                torch.cat(
                    [data_loader.dataset.item_id.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
            )
            ratingCat = (
                torch.cat(
                    [data_loader.dataset.rating.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
            )

        else:
            userID = data_loader.dataset.user_id.clone().to(device)
            itemID = data_loader.dataset.item_id.clone().to(device)
            ratingCat = data_loader.dataset.rating.clone().to(device)

        loss = loss_fn(
            userID, itemID, ratingCat, predsTensor, targetItemTensor, y_v_toLoss
        )
    print("test_loss:", loss)

    return loss


def f_train(
    data_loader,
    model,
    loss_fn,
    optimizer,
    selected_items=[],
    selected_userIDs=[],
    pretrain=False,
):
    model.train()
    all_preds = []
    target_preds = []
    target_item = torch.full((BATCH_SIZE,), fill_value=TARGET).to(device)
    n_records = len(data_loader) * BATCH_SIZE
    y_v = []
    y_v_toLoss = torch.tensor([0.0]).clone().to(device)
    additionalZeros = 0
    for user, item, rating in data_loader:
        X_user = torch.from_numpy(np.array(user)).clone().to(device)
        X_item = torch.from_numpy(np.array(item)).clone().to(device)
        if len(X_user) < BATCH_SIZE:
            additionalZeros = (
                torch.full((BATCH_SIZE - len(X_user),), fill_value=0).clone().to(device)
            )
            additionalZerosAlt = (
                torch.full((BATCH_SIZE - len(X_user), 1), fill_value=0)
                .clone()
                .to(device)
            )
            target_itemAlt = torch.full(
                (BATCH_SIZE - (BATCH_SIZE - len(X_user)),), fill_value=TARGET
            ).to(device)
            pred = model(X_user, X_item).to(device)
            pred = torch.cat([pred, additionalZerosAlt], dim=0).clone().to(device)
            all_preds.append(pred)
            t_predictions = model(X_user, target_itemAlt).clone().to(device)
            t_predictions = (
                torch.cat([t_predictions, additionalZerosAlt], dim=0).clone().to(device)
            )
            target_preds.append(t_predictions)
        else:
            pred = model(X_user, X_item).to(device)
            all_preds.append(pred)
            target_preds.append(model(X_user, target_item).clone().to(device))
    if n_users > 944:
        X_user_v = torch.from_numpy(np.array(selected_userIDs)).clone().to(device)
        X_item_v = torch.from_numpy(np.array(selected_items)).clone().to(device)
        y_v_toLoss = model(X_user_v, X_item_v).clone().to(device)

    predsTensor = torch.reshape(torch.stack(all_preds, dim=-1), (n_records,)).to(device)
    targetItemTensor = torch.reshape(
        torch.stack(target_preds, dim=-1), (n_records,)
    ).to(device)

    if pretrain == True:
        if type(additionalZeros) == torch.Tensor:
            loss = loss_fn(
                predsTensor,
                torch.cat(
                    [data_loader.dataset.rating.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
                .detach(),
            )
        else:
            print(
                n_records,
                data_loader.dataset.rating.size(),
                len(data_loader),
                predsTensor.size(),
            )
            loss = loss_fn(
                predsTensor, data_loader.dataset.rating.clone().to(device).detach()
            )
    else:
        if type(additionalZeros) == torch.Tensor:

            userID = (
                torch.cat(
                    [data_loader.dataset.user_id.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
            )
            itemID = (
                torch.cat(
                    [data_loader.dataset.item_id.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
            )
            ratingCat = (
                torch.cat(
                    [data_loader.dataset.rating.to(device), additionalZeros], dim=0
                )
                .clone()
                .to(device)
            )

        else:
            userID = data_loader.dataset.user_id.clone().to(device)
            itemID = data_loader.dataset.item_id.clone().to(device)
            ratingCat = data_loader.dataset.rating.clone().to(device)

        loss = loss_fn(
            userID, itemID, ratingCat, predsTensor, targetItemTensor, y_v_toLoss
        )
    print("train_loss:", loss)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def get_predictions(prediction_model, user_li, item_li, rating_li):
    prediction_model.eval()

    df_for_pred = pd.DataFrame(
        data={
            "userID": user_li,
            "itemID": item_li,
            "rating": rating_li,
        }
    )

    pred_dataset = MLDataset(df_for_pred)
    pred_dataloader = DataLoader(pred_dataset, batch_size=BATCH_SIZE)

    n_records = (len(pred_dataloader) - 1) * BATCH_SIZE

    preds = []

    for user, item, rating in pred_dataloader:
        if len(user) == BATCH_SIZE:
            X_user = torch.from_numpy(np.array(user))
            X_item = torch.from_numpy(np.array(item))
            single_pred = prediction_model(X_user, X_item)
            preds.append(single_pred)
        else:
            X_user = torch.from_numpy(np.array(user))
            X_item = torch.from_numpy(np.array(item))
            last_batch = prediction_model(X_user, X_item)

    predsTensor = torch.reshape(torch.stack(preds, dim=-1), (n_records,))
    last_batch = torch.reshape(last_batch, (len(last_batch),))
    predsTensor = torch.cat([predsTensor, last_batch], dim=0).detach().numpy()

    final_pred_df = pd.DataFrame(
        data={"userID": user_li, "itemID": item_li, "rating": predsTensor}
    )

    return final_pred_df


if __name__ == "__main__":
    print("Device:", device)
    df = pd.read_csv(DATASET_PATH)
    n_users = df["userID"].nunique()
    n_items = df["itemID"].nunique()

    # Experiment data collection

    pretrain_train_loss_list = []
    pretrain_test_loss_list = []
    pretrain_hitratio_list = []
    pretrain_hitratio_t_list = []
    pretrain_auc_list = []
    poisoned_train_loss_list = []
    poisoned_test_loss_list = []
    poisoned_hitratio_list = []
    poisoned_hitratio_t_list = []
    poisoned_auc_list = []
    target_item_log_list = []
    poisoned_test_hitratio = []
    pretrain_test_hitratio = []
    n_model_params = 0

    # unpopularItem Selection
    unpopularItems = df.groupby("itemID").agg("count")[
        df.groupby("itemID").agg("count")["userID"] < 6
    ]
    unpopularItemList = [
        i for i in random.sample(list(unpopularItems.index), n_target_items)
    ]

    # negative item creation

    negative_userID = []
    negative_itemID = []
    negative_rating = []
    negative_timestamp = []

    min_timestamp = min(df["timestamp"])
    max_timestamp = max(df["timestamp"])

    all_items = [i for i in range(1, n_items + 1)]
    all_timestamps = set(df["timestamp"].values)
    total_negative_items = num_negative_items * (n_users)

    for i in range(1, n_users):
        user_item_selection = set(df[df["userID"] == i]["itemID"].values)
        negative_userID.extend([i for value in range(num_negative_items)])
        negative_itemID.extend(
            [
                list(set(all_items) - user_item_selection)[value]
                for value in random.sample(
                    range(1, n_items - len(user_item_selection)), num_negative_items
                )
            ]
        )
        negative_rating.extend([0.0 for value in range(num_negative_items)])
        negative_timestamp.extend(
            random.sample(range(min_timestamp, max_timestamp), num_negative_items)
        )

    negative_df = pd.DataFrame(
        data={
            "userID": negative_userID,
            "itemID": negative_itemID,
            "rating": negative_rating,
            "timestamp": negative_timestamp,
        }
    ).sample(frac=1, random_state=SEED)

    train, test = make_splits(df)
    negative_train, negative_test = make_splits(
        negative_df, min_n_items=num_negative_items / 2
    )
    train = pd.merge(train, negative_train, how="outer")
    test = pd.merge(test, negative_test, how="outer")

    orig_train = train
    orig_test = test

    print("Trainset length:", len(train))
    print("Testset length:", len(test))

    for t_item in unpopularItemList:

        TARGET = t_item

        train = orig_train
        test = orig_test

        target_item_log_list.append(TARGET)
        print("TARGET", TARGET)

        n_users = df["userID"].nunique()
        selection_prob = [1.0 for i in range(n_items)]
        selected_items = []
        y_v_userID = []
        for v in range(m):
            fake_user_ids = [944 + v]
            fake_item_ids = [TARGET]
            fake_timestamp = [888888888]
            fake_ratings = [1]
            fake_user_df = pd.DataFrame(
                data={
                    "userID": fake_user_ids,
                    "itemID": fake_item_ids,
                    "rating": fake_ratings,
                    "timestamp": fake_timestamp,
                }
            )
            train = pd.merge(train, fake_user_df, how="outer")
            n_users += 1

            train_dataset = MLDataset(train)
            test_dataset = MLDataset(test)
            train_dataloader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE, pin_memory=True
            )
            test_dataloader = DataLoader(
                test_dataset, batch_size=BATCH_SIZE, pin_memory=True
            )

            pretrain_loss_fn = nn.BCELoss()
            loss_fn = HeuristicHitRatio().to(device)
            model_pretrain = NeuralNetwork(n_factors, n_users, n_items).to(device)
            optimizer = torch.optim.Adam(model_pretrain.parameters(), lr=pretrain_lr)
            early_stopping = EarlyStopping(patience=3, verbose=True)
            if v == 0 and len(pretrain_train_loss_list) == 0:
                n_model_params = sum(
                    p.numel() for p in model_pretrain.parameters() if p.requires_grad
                )
                print("The number of params in model:", n_model_params)

            for epoch in range(PRETRAIN_EPOCHS):
                train_loss = f_train(
                    train_dataloader,
                    model_pretrain,
                    pretrain_loss_fn,
                    optimizer,
                    pretrain=True,
                    selected_items=selected_items,
                    selected_userIDs=y_v_userID,
                )
                test_loss = f_test(
                    test_dataloader,
                    model_pretrain,
                    pretrain_loss_fn,
                    pretrain=True,
                    selected_items=selected_items,
                    selected_userIDs=y_v_userID,
                )
                early_stopping(test_loss, model_pretrain)
                if early_stopping.early_stop:
                    break
            if v == 0:
                torch.save(torch.load(early_stopping.path), "./initial_model.pkl")
                print("saved initial model")
                pretrain_train_loss_list.append(
                    train_loss.cpu().detach().numpy().item()
                )
                pretrain_test_loss_list.append(test_loss.cpu().detach().numpy().item())
            optimizer = torch.optim.Adam(
                model_pretrain.parameters(), lr=poison_training_lr
            )
            early_stopping = EarlyStopping(patience=5, verbose=True)
            for epoch in range(POISON_EPOCHS):
                train_loss = f_train(
                    train_dataloader,
                    model_pretrain,
                    loss_fn,
                    optimizer,
                    pretrain=False,
                    selected_items=selected_items,
                    selected_userIDs=y_v_userID,
                )
                test_loss = f_test(
                    test_dataloader,
                    model_pretrain,
                    loss_fn,
                    pretrain=False,
                    selected_items=selected_items,
                    selected_userIDs=y_v_userID,
                )
                early_stopping(test_loss, model_pretrain)
                if early_stopping.early_stop:
                    break
            model_pretrain.load_state_dict(torch.load(early_stopping.path))

            # fake user の top-k items を求める
            fake_user_ids = [944 + v for i in range(n_items + 2)]
            fake_item_ids = [i + 1 for i in range(n_items + 1)]
            fake_item_ids.remove(TARGET)
            fake_timestamp = [888888888 for i in range(n_items + 1)]

            X_fake_user = torch.from_numpy(np.array(fake_user_ids)).clone().to(device)
            X_fake_item = torch.from_numpy(np.array(fake_item_ids)).clone().to(device)

            fake_all_preds = model_pretrain(X_fake_user, X_fake_item).to(device)
            fake_all_preds_df = pd.DataFrame(
                data={
                    "userID": fake_user_ids,
                    "itemID": fake_item_ids,
                    "rating": fake_all_preds.reshape(1, n_items - 2)
                    .cpu()
                    .detach()
                    .numpy()[0],
                    "timestamp": fake_timestamp,
                }
            )
            topk_all_preds_df = fake_all_preds_df.sort_values(
                "rating", ascending=False
            ).head(TOP_K)

            altered_probs = []
            altered_item_ids = []
            altered_user_ids = []
            altered_timestamp = []
            topk_item_ids = topk_all_preds_df["itemID"].values

            # Better:row-wiseな処理より速い方法は多分ある
            for i, j in fake_all_preds_df.iterrows():
                altered_user_ids.append(int(j[0]))
                altered_item_ids.append(int(j[1]))
                altered_timestamp.append(int(j[3]))
                if int(j[1]) in topk_item_ids:
                    altered_probs.append(selection_prob[int(j[1])] * j[2])
                    selection_prob[int(j[1])] = selection_prob[int(j[1])] * delta
                else:
                    altered_probs.append(j[2])

            new_df = (
                pd.DataFrame(
                    data={
                        "userID": altered_user_ids,
                        "itemID": altered_item_ids,
                        "rating": altered_probs,
                        "timestamp": altered_timestamp,
                    }
                )
                .sort_values("rating", ascending=False)
                .head(TOP_K)
            )
            print(new_df)
            new_df["rating"] = new_df["rating"] * 5
            new_df["rating"] = new_df["rating"].astype(int)
            train = pd.merge(train, new_df, how="outer")

            for i in new_df["itemID"]:
                selected_items.append(i)
                y_v_userID.append(944 + v)

            if v == m - 1:
                train_dataset = MLDataset(train)
                test_dataset = MLDataset(test)
                train_dataloader = DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, pin_memory=True
                )
                test_dataloader = DataLoader(
                    test_dataset, batch_size=BATCH_SIZE, pin_memory=True
                )
                model_after_poison = NeuralNetwork(n_factors, n_users, n_items).to(
                    device
                )
                final_train_loss_fn = nn.BCELoss()
                final_optimizer = torch.optim.Adam(
                    model_after_poison.parameters(), lr=pretrain_lr
                )
                early_stopping = EarlyStopping(patience=3, verbose=True)
                for epoch in range(PRETRAIN_EPOCHS):
                    train_loss = f_train(
                        train_dataloader,
                        model_after_poison,
                        final_train_loss_fn,
                        final_optimizer,
                        pretrain=True,
                        selected_items=selected_items,
                        selected_userIDs=y_v_userID,
                    )
                    test_loss = f_test(
                        test_dataloader,
                        model_after_poison,
                        final_train_loss_fn,
                        pretrain=True,
                        selected_items=selected_items,
                        selected_userIDs=y_v_userID,
                    )
                    early_stopping(test_loss, model_after_poison)
                    if early_stopping.early_stop:
                        poisoned_train_loss_list.append(
                            train_loss.cpu().detach().numpy().item()
                        )
                        poisoned_test_loss_list.append(
                            test_loss.cpu().detach().numpy().item()
                        )
                        break
                torch.save(torch.load(early_stopping.path), "./poisoned_model.pkl")
                del fake_all_preds
                del model_after_poison
                del model_pretrain
                torch.cuda.empty_cache()
                gc.collect()

        # all_preds matrixの生成
        HR_user_ids = []
        HR_item_ids = []
        HR_ratings = []
        ratings_before_poison = []
        ratings_after_poison = []

        num_users = 943 + m + 1

        for i in range(1, 943 + 1):
            for j in range(1, n_items):
                HR_user_ids.append(i)
                HR_item_ids.append(j)
                HR_ratings.append(0.0)

        model_before_poison = NeuralNetwork(n_factors, num_users - m + 1, n_items)
        model_before_poison.load_state_dict(torch.load("initial_model.pkl"))

        all_preds_normal = get_predictions(
            model_before_poison, HR_user_ids, HR_item_ids, HR_ratings
        )

        model_after_poison = NeuralNetwork(n_factors, num_users, n_items)
        model_after_poison.load_state_dict(torch.load("poisoned_model.pkl"))
        all_preds_poisoned = get_predictions(
            model_after_poison, HR_user_ids, HR_item_ids, HR_ratings
        )

        del model_after_poison
        torch.cuda.empty_cache()

        normal_hits = 0
        poisoned_hits = 0

        for i in range(943 + 1):
            if (
                TARGET
                in all_preds_normal[all_preds_normal["userID"] == i]
                .sort_values("rating", ascending=False)
                .head(10)["itemID"]
                .values
            ):
                normal_hits += 1
        pretrain_hitratio_t_list.append(normal_hits / 943)
        print(normal_hits / 943)

        for i in range(943 + 1):
            if (
                TARGET
                in all_preds_poisoned[all_preds_poisoned["userID"] == i]
                .sort_values("rating", ascending=False)
                .head(10)["itemID"]
                .values
            ):
                poisoned_hits += 1
        poisoned_hitratio_t_list.append(poisoned_hits / 943)
        print(poisoned_hits / 943)

        accuracy_hit = 0
        for i in range(1, 943 + 1):
            for j in (
                all_preds_normal[all_preds_normal["userID"] == i]
                .sort_values("rating", ascending=False)
                .head(10)["itemID"]
                .values
            ):
                if j in df[df["userID"] == i]["itemID"].values:
                    accuracy_hit += 1

        pretrain_hitratio_list.append(accuracy_hit / 943)
        print("normal preds hit ratio:", accuracy_hit / 943)

        accuracy_hit = 0
        for i in range(1, 943 + 1):
            for j in (
                all_preds_poisoned[all_preds_poisoned["userID"] == i]
                .sort_values("rating", ascending=False)
                .head(10)["itemID"]
                .values
            ):
                if j in df[df["userID"] == i]["itemID"].values:
                    accuracy_hit += 1

        poisoned_hitratio_list.append(accuracy_hit / 943)
        print("poisoned preds hit ratio:", accuracy_hit / 943)

        ac_test = test
        ac_test["rating"] = ac_test["rating"].astype(bool).astype("f")
        ac_test["userID"] = ac_test["userID"].astype(int)
        ac_test["itemID"] = ac_test["itemID"].astype(int)

        num_users = 943 + m + 1

        model_before_poison = NeuralNetwork(n_factors, num_users - m + 1, n_items)
        model_before_poison.load_state_dict(torch.load("initial_model.pkl"))
        preds_normal = get_predictions(
            model_before_poison,
            ac_test["userID"].values,
            ac_test["itemID"].values,
            ac_test["rating"].values,
        )
        pretrain_auc = roc_auc_score(
            ac_test["rating"].values, preds_normal["rating"].values
        )
        pretrain_auc_list.append(pretrain_auc)
        print("normal auc:", pretrain_auc)

        del preds_normal
        del model_before_poison

        model_before_poison = NeuralNetwork(n_factors, num_users, n_items)
        model_before_poison.load_state_dict(torch.load("poisoned_model.pkl"))
        preds_normal = get_predictions(
            model_before_poison,
            ac_test["userID"].values,
            ac_test["itemID"].values,
            ac_test["rating"].values,
        )

        poisoned_auc = roc_auc_score(
            ac_test["rating"].values, preds_normal["rating"].values
        )

        poisoned_auc_list.append(poisoned_auc)
        print("poisoned auc:", poisoned_auc)

        hit_train, hit_test = make_splits(df)
        hit_test["rating"] = hit_test["rating"].astype(bool).astype("f")
        all_items = set([i for i in range(1, n_items - 1)])
        acc_user = []
        acc_item = []
        num_users = 943 + m + 1

        for i in range(1, 943 + 1):
            for j in set(
                all_items - set(hit_train[hit_train["userID"] == i]["itemID"].values)
            ):
                acc_user.append(i)
                acc_item.append(j)

        model_before_poison = NeuralNetwork(n_factors, num_users - m + 1, n_items)
        model_before_poison.load_state_dict(torch.load("initial_model.pkl"))
        all_preds_normal = get_predictions(
            model_before_poison, acc_user, acc_item, [0.0 for i in range(len(acc_user))]
        )

        del preds_normal
        del model_before_poison

        num_hits = 0

        for i in range(943 + 1):
            if (
                len(
                    set(hit_test[hit_test["userID"] == i]["itemID"].values)
                    & set(
                        all_preds_normal[all_preds_normal["userID"] == i]
                        .sort_values("rating", ascending=False)
                        .head(10)["itemID"]
                        .values
                    )
                )
                > 0
            ):
                num_hits += 1
        pretrain_test_hitratio.append(num_hits / 943)
        print(num_hits / 943)

        num_hits = 0

        model_after_poison = NeuralNetwork(n_factors, num_users, n_items)
        model_after_poison.load_state_dict(torch.load("poisoned_model.pkl"))
        all_preds_poisoned = get_predictions(
            model_after_poison, acc_user, acc_item, [0.0 for i in range(len(acc_user))]
        )

        for i in range(943 + 1):
            if (
                len(
                    set(hit_test[hit_test["userID"] == i]["itemID"].values)
                    & set(
                        all_preds_poisoned[all_preds_poisoned["userID"] == i]
                        .sort_values("rating", ascending=False)
                        .head(10)["itemID"]
                        .values
                    )
                )
                > 0
            ):
                num_hits += 1
        poisoned_test_hitratio.append(num_hits / 943)
        print(num_hits / 943)

        # del preds_poison
        del model_after_poison
        torch.cuda.empty_cache()

    experiment_result_df = pd.DataFrame(
        data={
            "target_item_id": target_item_log_list,
            "pretrain_hitratio@10": pretrain_hitratio_list,
            "poisoned_hitratio@10": poisoned_hitratio_list,
            "pretrain_hitratio@t": pretrain_hitratio_t_list,
            "poisoned_hitratio@t": poisoned_hitratio_t_list,
            "pretrain_train_loss": pretrain_train_loss_list,
            "pretrain_test_loss": pretrain_test_loss_list,
            "pretrain_auc": pretrain_auc_list,
            "poisoned_auc": poisoned_auc_list,
            "pretrain_hitratio(test)": pretrain_test_hitratio,
            "poisoned_hitratio(test)": poisoned_test_hitratio,
        }
    )

    experiment_result_df.to_csv(
        "exp_result" + str(n_model_params) + ".csv", index=False
    )
