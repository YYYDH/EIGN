import torch.optim as optim
import pandas as pd
import argparse
from utils import AverageMeter
from EIGN import GIN
from dataset_onefeat import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
from utils import *
from sklearn.metrics import mean_squared_error

from torch.utils.tensorboard import SummaryWriter


def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data = data.to(device)
        # x, edge_index_inter, pos = \
        #     data.x, data.edge_index_inter, data.pos
        with torch.no_grad():
            # x_inter = model_en.encode(x, edge_index_inter)
            pred = model(data)
            # pred = model(data, x_inter)
            label = data.y

            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff


if __name__ == '__main__':
    cfg = 'TrainConfig_GIGN'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    early_stop_epoch = args.get("early_stop_epoch")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--set', type=str, default='general')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=800)
    argsp = parser.parse_args()

    for repeat in range(argsp.repeat):
        args['repeat'] = repeat

        test2013_dir = os.path.join(data_root, 'test2013')
        test2016_dir = os.path.join(data_root, 'test2016')

        test2013_df = pd.read_csv(os.path.join(data_root, 'test2013_195.csv'))
        test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))

        test2013_dataset = 'test2013_195'
        test2016_dataset = 'test2016'

        if argsp.set == 'general':
            train_dataset = 'train_general'
            valid_dataset = 'valid_general'

            train_dir = os.path.join(data_root, 'train_general')
            valid_dir = os.path.join(data_root, 'valid_general')
            train_df = pd.read_csv(os.path.join(data_root, 'train_general.csv'))
            valid_df = pd.read_csv(os.path.join(data_root, 'valid_general.csv'))

        elif argsp.set == 'refined':
            train_dataset = 'train_refined'
            valid_dataset = 'valid_refined'

            train_dir = os.path.join(data_root, 'train_refined')
            valid_dir = os.path.join(data_root, 'valid_refined')
            train_df = pd.read_csv(os.path.join(data_root, 'train_refined.csv'))
            valid_df = pd.read_csv(os.path.join(data_root, 'valid_refined.csv'))

        train_set = GraphDataset(train_dir, train_df, data_root, train_dataset, create=True)
        valid_set = GraphDataset(valid_dir, valid_df, data_root, valid_dataset, create=True)
        test2013_set = GraphDataset(test2013_dir, test2013_df, data_root, test2013_dataset,
                                    create=True)
        test2016_set = GraphDataset(test2016_dir, test2016_df, data_root, test2016_dataset,
                                    create=True)

        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                                     persistent_workers=True)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=1,
                                     pin_memory=True, persistent_workers=True)
        test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=1,
                                        pin_memory=True, persistent_workers=True)
        test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=1,
                                        pin_memory=True, persistent_workers=True)

        csar_dir = os.path.join(data_root, 'csar')
        csar_dataset = 'csar'
        csar_df = pd.read_csv(os.path.join(data_root, 'csar.csv'))
        csar_set = GraphDataset(csar_dir, csar_df, data_root, csar_dataset, create=True)
        csar_loader = PLIDataLoader(csar_set, batch_size=batch_size, shuffle=False, num_workers=1,
                                     pin_memory=True, persistent_workers=True)

        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")

        device = torch.device('cuda:' + argsp.cuda if torch.cuda.is_available() else "cpu")

        in_drop_rate = 0.1
        out_drop_rate = 0.1
        model = GIN(35, 256, in_drop_rate=in_drop_rate, out_drop_rate=out_drop_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
        criterion = nn.MSELoss()

        writer = SummaryWriter(logger.get_model_dir())

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []

        model.train()
        print(argsp.epochs)
        for epoch in range(argsp.epochs):
            for data in train_loader:
                data = data.to(device)
                pred = model(data)
                label = data.y

                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0))

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            writer.add_scalar('train/loss', epoch_loss, epoch)

            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            # scheduler.step(valid_rmse)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)
            writer.add_scalar('valid/rmse', valid_rmse, epoch)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                          % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), 'best_model.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        writer.close()

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test2013_rmse, test2013_pr = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)
        csar_rmse, csar_pr = val(model, csar_loader, device)

        msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, csar_rmse-%.4f, csar_pr-%.4f," \
              % (
                  valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, csar_rmse, csar_pr)

        logger.info(msg)
