import torch
import os
import logging
from argparse import ArgumentParser
from src.utils.dataloaders import load_mnist_dataloader, load_image_folder_dataloader
from src.models.classification import CLSModel
from src.utils.decoders import softmax_decoder
from src.experiment.classification import ClassificationExperiment
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4
from src.utils.parameters import write_params_to_file, load_parameters, instanciate_cls

if __name__ == "__main__":

    logging_message = "[AROB-2025-KAPTIOS]"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser()
    parser.add_argument("--params", type=str, default='./params/mnist.yaml')
    args = parser.parse_args()
    params = load_parameters(args.params)

    xp_params = params['experiment']['parameters']
    data_params = params['dataset']['parameters']
    net_params = params['network']['parameters']
    enc_params = params['encoder']['parameters']

    ## ========== INIT ========== ##

    gpu = torch.cuda.is_available()

    if gpu:
        torch.cuda.manual_seed_all(xp_params["seed"])
    else:
        torch.manual_seed(xp_params["seed"])

    experiment_id = str(uuid4())[:8]
    experiment_name = f'experiment_{experiment_id}' if not params['experiment'][
        'name'] else f"{params['experiment']['name']} - {experiment_id}"
    logging.info(
        'Initialization of the experiment protocol - {}'.format(experiment_name))
    log_dir = os.path.join(xp_params["log_dir"], experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    write_params_to_file(vars(args), log_dir)
    log_interval = max(100 // data_params["batch_size"], 1)

    # ========== DATALOADER ========== ##

    if params['dataset']['name'] == 'MNIST':
        train_dl, test_dl, n_classes = load_mnist_dataloader(
            data_params["data_dir"],
            data_params["resize"],
            data_params["batch_size"],
            gpu)
    else:
        train_dl, test_dl, n_classes = load_image_folder_dataloader(
            data_params["data_dir"],
            data_params["resize"],
            data_params["batch_size"],
            gpu)

    logging.info('Dataloaders successfully loaded.')

    ## ========== MODEL ========== ##

    DEVICE = torch.device("cuda") if gpu else torch.device("cpu")
    input_encoder = instanciate_cls(
        params['encoder']['module'], params['encoder']['name'], enc_params)

    net = instanciate_cls('src.networks.classification',
                          params['network']['name'], net_params)

    model = CLSModel(
        encoder=input_encoder,
        snn=net,
        decoder=softmax_decoder
    ).to(DEVICE)

    # ========== TRAINING ========== ##

    writer = SummaryWriter(log_dir=log_dir, flush_secs=60)

    logging.info(f'Running on device : {DEVICE}')
    experiment = ClassificationExperiment(
        model=model,
        writer=writer,
        lr=xp_params["lr"],
        log_interval=log_interval,
        device=DEVICE)

    experiment.fit(train_dl, test_dl, xp_params["epochs"])
