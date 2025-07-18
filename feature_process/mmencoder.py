import torch
import clip # 环境里没有这个包，记得下
from torchvision import transforms
import glob
import pickle
from PIL import Image
import os
import clip


class ImgTextEncoder:
    def __init__(self, clip_model_path, data_path):
        self.clip_model_path = "../../CLIP"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root=clip_model_path)
        self.data_path = data_path

    def text_encoder(self, text_file_path, entity2id_path, max_length=77,
                     save_path="../data/WN9/embeddings/text_features.pkl"):
        # 加载实体ID顺序
        with open(entity2id_path, 'r') as f:
            entity2id = {line.split('\t')[0]: int(line.split('\t')[1].strip()) for line in f.readlines()}

        texts = []
        entity_ids = []
        # 准备文本数据
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            for line in text_file:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    entity_id = parts[0]
                    if entity_id in entity2id:
                        entity_text = ' '.join(parts[1:])
                        texts.append(entity_text)
                        entity_ids.append(entity_id)

        # 处理文本
        entity_text_features = {}
        for i, text in enumerate(texts):
            # 分割长文本为较短的片段
            text_segments = [text[j:j + max_length] for j in range(0, len(text), max_length)]
            segment_features = []
            for segment in text_segments:
                text_input = clip.tokenize([segment]).to(self.device)
                with torch.no_grad():
                    segment_feature = self.model.encode_text(text_input)
                segment_features.append(segment_feature)

            # 聚合特征向量，并去除额外的维度
            aggregated_feature = torch.mean(torch.stack(segment_features), dim=0)
            entity_text_features[entity_ids[i]] = aggregated_feature.squeeze().cpu().numpy()  # Flatten to (512,)

        # 保存特征
        with open(save_path, 'wb') as f:
            pickle.dump(entity_text_features, f)

        print(f"Saved text features to {save_path}")

if __name__ == "__main__":

    clip_model_path = "../../CLIP"  # 根据实际情况调整路径
    data_path = "../data/WN9"  # 根据实际情况调整路径
    encoder = ImgTextEncoder(clip_model_path, data_path)
    # 处理文本特征
    text_file_path = os.path.join(data_path, "_data_/gloss.txt")
    entity2id_path = os.path.join(data_path, "_data_/entity2id.txt")
    encoder.text_encoder(text_file_path, entity2id_path)

'''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformation_model = transforms.Compose([
    transforms.Resize((224,224), interpolation=0),
    transforms.ToTensor(),
    normalize
])

class ImgTextEncoder:
    def __init__(self, clip_model_path, data_path):
        self.clip_model_path = "../../CLIP"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root=clip_model_path)
        self.data_path = data_path

    def process_batch(self, input_img, entity):
        print(f"Processing images for entity {entity}.")
        if len(input_img) == 0:
            print("No images to process for this entity.")
            return
        # Stack and process all images for this entity
        batch_tensor = torch.stack(input_img, dim=0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor)
        # Calculate the average feature vector for this entity
        avg_feature = torch.mean(image_features, dim=0, keepdim=True)
        save_path = os.path.join("../data/WN9/embeddings/img_embeddings", entity, "avg_embedding.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb+") as f:
            pickle.dump(avg_feature.cpu().numpy(), f)
        print(f"Saved average feature for {entity}.")
    def img_encoder(self, batch_size = 32):  #batch_size表示的是一个实体的所有图像被视作一个批次，这样做的好处是避免内存不足，分实体按批次处理图像

        entity_ids = []
        input_img = []
        with open(os.path.join(self.data_path, "_data_/entity2id.txt"), "r") as enidf:

            for line in enidf:
                entity = line.strip().split('\t')[0]
                print(entity)
                entity_folder = f"{self.data_path}/ima_data/wn9_filtered/{entity}"
                print(f"Processing entity: {entity}")
                print(f"Looking for images in: {entity_folder}")
        base_path = os.path.join(self.data_path, "ima_data/wn9_filtered")
        entities = os.listdir(base_path)

        for entity in entities:
            entity_folder = os.path.join(base_path, entity)
            input_img = []
            #print(f"Looking for images in: {entity_folder}")
            for filename in glob.glob(entity_folder + "/*.jpg") + glob.glob(entity_folder + "/*.JPG") + glob.glob(
                    entity_folder + "/*.jpeg") + glob.glob(entity_folder + "/*.JPEG"):
                #print(filename)
                try:
                    im = Image.open(filename).convert("RGB")
                    im = transformation_model(im)
                    input_img.append(im)
                    #print(f"Loaded image: {filename}")
                    #entity_ids.append(entity)
                         #Process the batch if we reach the batch limit or if it's the last entity
                #    if len(input_img) >= batch_size or entity == entity_ids[-1]:
                    #if len(input_img) >= batch_size:
                        #self.process_batch(input_img, entity_ids)
                        #input_img = []
                        #entity_ids = []
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                        #print(f"Error processing {filename}: {e}")

            self.process_batch(input_img, entity)

    



# Assuming you have defined your data_path somewhere
if __name__ == "__main__":

    clip_model_path = "../..CLIP"  # 根据实际情况调整路径
    data_path = "../data/WN9"  # 根据实际情况调整路径
    encoder = ImgTextEncoder(clip_model_path, data_path)
    # 处理图像特征
    encoder.img_encoder(batch_size=2)

    # 处理文本特征
    text_file_path = os.path.join(data_path, "_data_/gloss.txt")

    entity2id_path = os.path.join(data_path, "_data_/entity2id.txt")
    save_path_text_features = os.path.join(data_path, "embeddings/text_features.pt")
    encoder.text_encoder(text_file_path, entity2id_path, save_path=save_path_text_features)'''
