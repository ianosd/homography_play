from pentip_dataset import PenTipDataset
from matplotlib import pyplot as plt

data_folder = "labeled"
annotations_file = "labeled/annotations.csv"

dataset = PenTipDataset(annotations_file, data_folder)
print(f"Size of dataset: {len(dataset)}")

for i in range (len(dataset)):
    image, output = dataset[i]
    plt.subplot(121)
    plt.imshow(image._t.permute((1, 2, 0)))
    plt.subplot(122)
    plt.imshow(output)
    plt.show()
