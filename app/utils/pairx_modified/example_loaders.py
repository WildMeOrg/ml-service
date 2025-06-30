from transformers import AutoModel
import torchvision.transforms as transforms
import pandas as pd

def wildme_multispecies_miewid(device):
    ckpt_path = f"conservationxlabs/miewid-msv2"
        
    multispecies_model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True)
    multispecies_model.to(device)

    multispecies_img_transforms = transforms.Compose([transforms.Resize((440, 440)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])])

    return multispecies_model, (440,440), multispecies_img_transforms

def toy_df():
    df = pd.DataFrame()
    df["annot"] = ['cow_0_0', 'cow_0_1']
    df["name"] = ['cow_0', 'cow_0']
    df["image"] = [f"data/{annot}.jpg" for annot in df["annot"]]

    return df