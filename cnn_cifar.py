def main():
    CIFAR_DIR = 'cifar-10-batches-py/'

    def unpicked(file):
        import pickle
        with open(file, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
        return cifar_dict

    dirs = ['batches.meta', 'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'data_batch_6']
    all_data = [0, 1, 2, 3, 4, 5, 6]

    for i, direc in zip(all_data, dirs):
        all_data[i] = unpicked(CIFAR_DIR+direc)

    batch_meta = all_data[0]
    data_batch1 = all_data[1]
    data_batch2 = all_data[2]
    data_batch3 = all_data[3]
    data_batch4 = all_data[4]
    data_batch5 = all_data[5]
    data_batch6 = all_data[6]
    print(batch_meta)


if __name__ == '__main__':
    main()
