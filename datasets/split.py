import csv
import random


def split_data_set(filename, rel_training_size, rel_testing_size):
    training = []
    test = []
    header = []

    training_filename = filename.replace(".csv", "-training.csv")
    test_filename = filename.replace(".csv", "-test.csv")

    if rel_training_size < 0 or rel_testing_size < 0:
        print("ERROR, negative input")
        exit()
    if rel_testing_size + rel_training_size > 1.0001:
        print("ERROR, FRACTION BIGGER THAN 1")
        exit()

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        header_read = False
        for row in reader:
            if not header_read:
                header.append(row)
                header_read = True
                continue
            k = random.random()
            if k <= 0.9:
                training.append(row)
            else:
                test.append(row)

    with open(training_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header[0])
        for i in training:
            writer.writerow(i)

    with open(test_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header[0])
        for i in test:
            writer.writerow(i)


split_data_set("breast-cancer.csv", 0.9, 0.1)
