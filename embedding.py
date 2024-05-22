import torch
from sentence_transformers import SentenceTransformer, util 
import pandas as pd
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

file_path = "Llava_report - Llava_1.6_34b.csv"
df = pd.read_csv(file_path)

shape_embedds = np.empty((0, 384), dtype=np.float32)
texture_embedds = np.empty((0, 384), dtype=np.float32)
text_embedds = np.empty((0, 384), dtype=np.float32)
color_embedds = np.empty((0, 384), dtype=np.float32)
file_embedding = np.empty((0, 384), dtype=np.float32)
first_embedding = np.empty((0, 384), dtype=np.float32)

mentioned_img_row=22
print(df.iloc[mentioned_img_row, 3])
first_shape_embedd = model.encode(df.iloc[mentioned_img_row, 1])
first_texture_embedd = model.encode(df.iloc[mentioned_img_row, 2])
first_text_embedd = model.encode(df.iloc[mentioned_img_row, 3])
first_color_embedd = model.encode(df.iloc[mentioned_img_row, 4])
first_embedding = np.vstack((first_embedding, first_shape_embedd))
first_embedding = np.vstack((first_embedding, first_texture_embedd))
first_embedding = np.vstack((first_embedding, first_text_embedd))
first_embedding = np.vstack((first_embedding, first_color_embedd))
# print(first_embedding.shape)

for i in range(len(df)):
    row_embedding = []
    if i==mentioned_img_row:
        continue
    for j in range(1, len(df.columns)):
        col_embeddings = model.encode(df.iloc[i, j])
        row_embedding.append(col_embeddings)
    row_embedds = np.array(row_embedding, dtype=np.float32)
    shape_embedds = np.vstack((shape_embedds, row_embedds[0]))
    texture_embedds = np.vstack((texture_embedds, row_embedds[1]))
    text_embedds = np.vstack((text_embedds, row_embedds[2]))
    color_embedds = np.vstack((color_embedds, row_embedds[3]))
    file_embedding = np.vstack((file_embedding, row_embedds))

cos_sim_shape = util.cos_sim(shape_embedds, first_shape_embedd)
cos_sim_texture = util.cos_sim(texture_embedds, first_texture_embedd)
cos_sim_text = util.cos_sim(text_embedds, first_text_embedd)
cos_sim_color = util.cos_sim(color_embedds, first_color_embedd)

# print(cos_sim_color)


final_cosine_similarity = []

# print(cos_sim_shape.shape)
for i in range(len(cos_sim_shape)-1):
    cos_score= (cos_sim_shape[i]+cos_sim_texture[i]+cos_sim_text[i]+cos_sim_color[i])/4
    final_cosine_similarity.append(cos_score)
    # print(cos_score)
# print(final_cosine_similarity)
     
all_sentence_combinations=[]
all_sentence_combinations = [(score, i) for i, score in enumerate(final_cosine_similarity) if i != mentioned_img_row]
print(all_sentence_combinations)

sorted_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

# Print top 5 most similar pairs
print("Top-5 most similar pairs:")
for score, i in sorted_combinations[:5]:
    print("Cosine Similarity:", score)
    print("Row Index:", i)
    print("text : ",df.iloc[i, 3])
   
    # print("Image name : ", df[i+1][0])
    # print("text : ",df[i][3])


# import json

# def add_prefix_to_filenames(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
    
#     for item in data:
#         filename = item.get('filename', '')  # Get the value of 'filename' key
#         if filename:
#             item['filename'] = f"journal_no_2153_{filename}"  # Prepend 'journal_no_2151_' to the filename

#     # Write the updated data back to the JSON file
#     with open(json_file_path, 'w') as f:
#         json.dump(data, f, indent=4)

# def main():
#     json_file_path = '/media/aditya/Projects/Logo_Matching/journal_watch/LLaVA/journal_no_2153.json'  # Replace with the path to your JSON file
#     add_prefix_to_filenames(json_file_path)

# if __name__ == "__main__":
#     main()
