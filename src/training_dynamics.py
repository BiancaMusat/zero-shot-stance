import re
import argparse

def generate_dynamics(logits_file, EPOCHS, BATCH_SIZE):
    """
    Function that generates the training dynamics files
    logits_file has the following format:
        - first line: all data ids in the order they appear into the results
        - for each epoch and each batch repeat:
            - epoch number
            - batch number
            - gold labels for the examples corresponding to the current epoch/batch
            - the loggits corresponding to the current epoch/batch
    EPOCHS is the number of epochs
    BATCH_SIZE is the sixe of the batch
    """
    all_data = []  # will contain all data from logits file
    with open(logits_file, "r") as f:
        for line in f:
            all_data.append(line)
    all_ids = all_data[0][1:-2]
    all_ids = [int(x) for x in all_ids.split(',')]  # get all examples' ids in order
    all_ids_b = []  # list of ids split into batches
    start = 0
    end = len(all_ids)
    step = BATCH_SIZE  # batch size
    for i in range(start, end, step):
        x = i
        all_ids_b.append(all_ids[x:x+step])
    all_data = all_data[1:]  # remove the ids from the data
    file_data = {x : [] for x in range(0, EPOCHS)}  # the logits corresponding to each epoch
    k = 0
    for i in range(0, len(all_data), 4):
        epoch = int(all_data[i])  # read the epoch
        batch = int(all_data[i + 1])  # read the batch number
        if batch == 0:
            k = 0
        labels = [int(x) for x in all_data[i + 2][1:-2].split(',')]  # read the gold labels
        all_logits = re.findall('\[.*?\]',all_data[i + 3][1:-2])  # read the logits
        logits = []
        for l in all_logits:
            logits.append([float(x) for x in l[1:-1].split(',')])  # generate the list of logits
        for j, id in enumerate(all_ids_b[k]):
                file_data[epoch].append({"guid" : id, "logits_epoch_"+str(epoch) : logits[j], "gold": labels[j]})  # enerate trainin dynamics
        k += 1

    file_names = ["training_dynamics/dynamics_epoch_" + str(x) + ".jsonl" for x in range(0, EPOCHS)]
    for i, file_name in enumerate(file_names):
        with open(file_name, "w") as f:  # write the dynamics to the corresponding files
            for data in file_data[i]:
                f.write(str(data).replace('\'', '\"') + '\n')  # change single quotes to double quotes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--logits_file', help='Name of the logits file', required=False)
    parser.add_argument('-e', '--epochs', help='The number of epochs', required=False)
    parser.add_argument('-b', '--batch_size', help='The number of batches', required=False)
    args = vars(parser.parse_args())

    logits_file = "logits.txt"
    epochs = 10
    batch_size = 64

    if args["logits_file"] is not None:
        logits_file = args["logits_file"]

    if args["epochs"] is not None:
        epochs = int(args["epochs"])

    if args["batch_size"] is not None:
        batch_size = int(args["batch_size"])

    generate_dynamics(logits_file, epochs, batch_size)
    