import openai

openai.api_key = "your-api-key"

english_labels = ["antrum", "body", "fundus", "cardia", "angularis"]

def translate_labels(labels):
    prompt = f"请将以下胃镜解剖部位的英文标签翻译为准确的中文：\n{', '.join(labels)}\n输出格式为'英文: 中文'"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

translations = translate_labels(english_labels)
print(translations)



# import json
# from collections import defaultdict
#
# json_files = [
#     'gastrohun-image-metadata.json',
#     'gastrohun-sequence-metadata.json',
#     'gastrohun-videoendoscopy-metadata.json'
# ]
#
# labels = defaultdict(set)
#
# # 提取标签
# for file in json_files:
#     with open(file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#         for item in data:
#             for key, value in item.items():
#                 labels[key].add(value)
#
# # 保存提取的唯一标签
# with open('unique_labels.json', 'w', encoding='utf-8') as f:
#     json.dump({k: list(v) for k, v in labels.items()}, f, ensure_ascii=False, indent=2)
#
# print('Unique labels extracted successfully!')
