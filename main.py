from data import dataset, data_loader

print(dataset[0][0].shape)

for x, y in data_loader:
    print(x.shape)
    break
