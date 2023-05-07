import datetime

import torch
import torch.nn as nn
import tqdm

from datasets import StockDataset, get_data, load_csv
from models import LSTM
from plot import loss_plot, tsne_plot
from settings import *

if __name__ == "__main__":
    train_series, test_series, n_features = get_data(
    )  # [time, stock, features], n_features
    n_stocks = train_series.shape[1]

    train_series = train_series.to(DEVICE)
    test_series = test_series.to(DEVICE)

    print(f"train_series.shape: {train_series.shape}")
    print(f"test_series.shape: {test_series.shape}")
    print(f"n_features: {n_features}")
    print(f"n_stocks: {n_stocks}")

    train_dataset = StockDataset(train_series, WINDOW_SIZE)
    test_dataset = StockDataset(test_series, WINDOW_SIZE)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    model = LSTM(
        sequence_length=WINDOW_SIZE,
        batch_size=BATCH_SIZE,
        input_size=n_features,
        output_size=n_features,
        num_embeddings=n_stocks,
        embedding_dim=EMBEDDING_SIZE,
        device=DEVICE,
    )

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print('Model parameters:')
    print(model)

    epoch_train_losses = []
    epoch_eval_losses = []

    for epoch in range(EPOCHS):
        # begin training
        model.train()
        pbar = tqdm.tqdm(train_dataloader)
        losses = []
        for i, sample in enumerate(pbar):
            X, Y_gt, stock_id = sample  # [batch_size, window_size, n_features], [batch_size, n_features], 1
            assert Y_gt is not None

            optimizer.zero_grad()
            model.clear_hidden_state(batch_size=BATCH_SIZE, device=DEVICE)

            X = X.to(DEVICE)
            Y_gt = Y_gt.to(DEVICE)
            stock_id = stock_id.to(DEVICE)

            Y_pred = model(X, stock_id)

            single_loss = loss_function(Y_pred, Y_gt)
            if single_loss > 100:
                print(f'loss is too large at {i}')
                print(f"X: {X}")
                print(f"Y_gt: {Y_gt}")
                print(f"Y_pred: {Y_pred}")
                print(f"stock_id: {stock_id}")
                continue

            if not torch.isnan(single_loss):
                single_loss.backward()
                optimizer.step()

            losses.append(single_loss.item())
            pbar.set_description(f'loss: {single_loss.item():10.8f}')

        epoch_train_losses.append(sum(losses) / len(losses))

        # begin evaluating
        with torch.no_grad():
            model.eval()
            losses = []
            pbar = tqdm.tqdm(test_dataloader)
            for i, sample in enumerate(pbar):
                X, Y_gt, stock_id = sample

                X = X.to(DEVICE)
                Y_gt = Y_gt.to(DEVICE)
                stock_id = stock_id.to(DEVICE)

                model.clear_hidden_state(batch_size=BATCH_SIZE, device=DEVICE)
                Y_pred = model(X, stock_id)
                single_loss = loss_function(Y_pred, Y_gt)
                losses.append(single_loss.item())
                pbar.set_description(f'loss: {single_loss.item():10.8f}')

            epoch_eval_losses.append(sum(losses) / len(losses))

    # plot
    loss_plot(epoch_eval_losses, "eval_loss.png")
    loss_plot(epoch_train_losses, "train_loss.png")
    if model.embedding.weight.device != 'cpu':
        tsne_plot(TICKER_LIST,
                  model.embedding.weight.detach().cpu().numpy(),
                  perplexity=PERPLEXITY)
    else:
        tsne_plot(TICKER_LIST,
                  model.embedding.weight.detach().numpy(),
                  perplexity=PERPLEXITY)

    # save model
    torch.save(model.state_dict(), "model.pt")

    # prediction
    data = load_csv(
        TICKER_LIST, FORMAT, PRED_START_DATE, PRED_START_DATE +
        datetime.timedelta(days=WINDOW_SIZE - 1))  # [time, stock, features]
    data = data.to(DEVICE)
    data = data.transpose(1, 0)  # [stock, time, features]
    stock_id = torch.arange(data.shape[0], dtype=torch.long, device=DEVICE)
    assert data.shape[
        1] == WINDOW_SIZE, f"data.shape[1] != {WINDOW_SIZE}, got {data.shape[1]}"

    pred_date = PRED_START_DATE + datetime.timedelta(days=WINDOW_SIZE)
    # pred_date_str = pred_date.strftime("%Y-%m-%d")
    pred_end_date = PRED_END_DATE

    print(f"pred_date: {pred_date}")
    print(f"pred_end_date: {pred_end_date}")
    # WARNING: off by 1 above?

    predictions = torch.empty((data.shape[0], 0, data.shape[2]),
                              device=DEVICE)  # [stock, time, features]
    with torch.no_grad():
        while pred_date <= pred_end_date:
            model.clear_hidden_state(batch_size=data.shape[0], device=DEVICE)

            Y_pred = model(data, stock_id)  # [stock, features]
            Y_pred = Y_pred.unsqueeze(1)  # [stock, 1, features]

            predictions = torch.cat((predictions, Y_pred),
                                    dim=1)  # [stock, time, features]
            data = torch.cat((data[:, 1:, :], Y_pred),
                             dim=1)  # [stock, time, features]
            pred_date += datetime.timedelta(days=1)
            # pred_date_str = pred_date.strftime("%Y-%m-%d")
    torch.save(predictions, "predictions.pt")
