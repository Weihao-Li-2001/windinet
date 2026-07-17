from windinet.training.shockwave_data import ShockWaveDataset


dataset = ShockWaveDataset(
    "sandbox/train.h5"
)


print(len(dataset))


sample = dataset[0]


for k,v in sample.items():

    if hasattr(v,"shape"):
        print(k, v.shape)

    else:
        print(k,v)