import torch.optim as optim
import pandas as pd
from EIGN import GIN
from dataset_onefeat import GraphDataset, PLIDataLoader
from config.config_dict import Config
from utils import *
from sklearn.metrics import mean_squared_error


def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    return rmse, coff


if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')

    # valid_dir = os.path.join(data_root, 'valid_general')
    test2013_dir = os.path.join(data_root, 'test2013')
    test2016_dir = os.path.join(data_root, 'test2016')
    # valid_dataset = 'valid2020'
    test2013_dataset = 'test2013_195'
    test2016_dataset = 'test2016'

    # valid_df = pd.read_csv(os.path.join(data_root, 'valid2020.csv'))
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013_195.csv'))
    test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))

    # valid_set = GraphDataset(valid_dir, valid_df, data_root, valid_dataset, create=True)
    test2013_set = GraphDataset(test2013_dir, test2013_df, data_root, test2013_dataset,
                                create=True)
    test2016_set = GraphDataset(test2016_dir, test2016_df, data_root, test2016_dataset,
                                create=True)

    # valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=1,
    #                              pin_memory=True, persistent_workers=True)
    test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=1,
                                    pin_memory=True, persistent_workers=True)
    test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=1,
                                    pin_memory=True, persistent_workers=True)

    device = torch.device('cuda:0')

    in_drop_rate = 0.1
    out_drop_rate = 0.1
    model = GIN(35, 256, in_drop_rate=in_drop_rate, out_drop_rate=out_drop_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    criterion = nn.MSELoss()

    model.load_state_dict(torch.load('./model/best_model1.pt', map_location=device))
    # valid_rmse, valid_pr = val(model, valid_loader, device, 'valid')
    test2013_rmse, test2013_pr = val(model, test2013_loader, device)
    test2016_rmse, test2016_pr = val(model, test2016_loader, device)

    msg = "test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f," \
          % (
              test2013_rmse, test2013_pr, test2016_rmse, test2016_pr)
    print(msg)


