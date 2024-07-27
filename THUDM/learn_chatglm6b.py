from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

tokenizer = AutoTokenizer.from_pretrained("F:\\learn\\AI\\carr001\\learn_ai\\third_party\\ChatGLM-6b\\THUDM\\chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("F:\\learn\\AI\\carr001\\learn_ai\\third_party\\ChatGLM-6b\\THUDM\\chatglm-6b", trust_remote_code=True).half().cuda()

model = model.eval()
#response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选哪个？", history=[],max_length=1)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=1)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=2)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=3)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=4)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=5)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=6)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=7)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=8)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=9)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=10)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=11)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=12)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=13)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=14)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=15)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=16)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=17)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=18)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=19)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=20)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=21)
response, history = model.chat(tokenizer, "下面是一个选择题：苹果是 A:动物，B：植物。选A还是选B？", history=[], max_new_tokens=22)
response, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(tokenizer, "你好", history=[])
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

device = model.device
# 加载数据集
dataset = load_dataset('cais/mmlu','all')
# 初始化指标列表
all_predictions = []
all_labels = []
# 设置评估模式
model.eval()
# 遍历数据集的评估部分
for batch in dataset['validation']:
    # 预处理数据
    inputs = tokenizer(batch['question'], padding=True, truncation=True, return_tensors='pt').to(device)

    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果
    predictions = torch.argmax(outputs.logits, dim=-1)

    # 与真实标签进行比较
    batch['prediction'] = predictions
    # 收集预测结果和真实标签
    all_predictions.extend(predictions)
    # all_labels.extend(batch['label'])

# 计算性能指标
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")