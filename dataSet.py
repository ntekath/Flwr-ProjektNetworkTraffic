import os
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader


def get_dataSet():
    # Pfade zu den Trainings- und Testdaten
    dataset_path = "C:/Users/noelt/OneDrive/Desktop/Studium/PraxisProjekt/Datensatz/archive"
    data_file = "Train_data.csv"
    train_path = os.path.join(dataset_path, data_file)
    test_path = os.path.join(dataset_path, data_file)

    # Lade die Trainingsdaten
    train_data = pd.read_csv(train_path, delimiter=",", header=0)

    # Entferne die kategorischen Spalten aus dem Trainingsdatensatz
    train_data_numeric = train_data.drop(columns=['protocol_type', 'service', 'flag'])

    # Konvertiere die Zielklasse in One-Hot-Encoding
    target_class = 'class'
    train_data_encoded = pd.get_dummies(train_data_numeric, columns=[target_class])

    # Extrahiere die Zielklasse für das Testset und entferne kategorische Spalten
    test_data = pd.read_csv(test_path, delimiter=",", header=0)

    # Entfernen der kategorischen spalten
    test_data_numeric = test_data.drop(columns=['protocol_type', 'service', 'flag'])
    test_data_encoded = pd.get_dummies(test_data_numeric, columns=[target_class])
    test_target = test_data[target_class]

    # Konvertiere die Zielklasse in numerischen Wert
    target_mapping = {'normal': 0, 'anomaly': 1}  # Beispiel für Mapping
    test_target_numeric = test_target.map(target_mapping)

    # Konvertiere alles in PyTorch-Tensoren
    train_tensor = torch.tensor(train_data_encoded.values.astype(float), dtype=torch.float32)
    test_tensor = torch.tensor(test_data_encoded.values.astype(float), dtype=torch.float32)
    test_target_tensor = torch.tensor(test_target_numeric.values.astype(float), dtype=torch.float32)

    return train_tensor, test_tensor, test_target_tensor


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    trainset, testset, test_target = get_dataSet()

    # Überprüfe die Gesamtlänge des Trainingsdatensatzes
    total_length = len(trainset)
    print("Gesamtlänge des Trainingsdatensatzes:", total_length)

    print("Gesamtlänge des Trainingsdatensatzes:", len(testset))

    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    num_images = total_length // num_partitions

    rest = total_length % num_partitions
    partition_len = [num_images + 1 if i < rest else num_images for i in range(num_partitions)]

    # Überprüfe die Summe der Partitionslängen
    print("Summe der Partitionslängen:", sum(partition_len))

    # split randomly
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []

    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # construct data loaders and append to their respective list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    testloader = DataLoader(testset, batch_size=128) # <- 128 = mat1 (128 x 40)

    return trainloaders, valloaders, testloader
