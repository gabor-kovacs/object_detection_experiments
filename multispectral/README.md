# Multispectral Example

This is an example of how to do object detection on special cases.

[Dataset](https://www.webpages.uidaho.edu/vakanski/Multispectral_Images_Dataset.html)

```bash
cd multispectral
wget https://www.webpages.uidaho.edu/vakanski/Codes_Data/Spectral_Images.zip
unzip Spectral_Images.zip
rm Spectral_Images.zip
```

First we prepare the dataset by creating aligned 4 channel tiffs from the individual channels.

Since the convenient ultralytics version will no longer work, let's use a pytorch implementation of rt-detr and modify it to work with our dataset.

git clone https://github.com/lyuwenyu/RT-DETR

intall the specific versions needed (older than the ultralytics version)

```bash
cd RT-DETR/rtdetr_pytorch
pip install -r requirements.txt
```

