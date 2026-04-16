import json
import glob
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 데이터로드
PATH = './TL_내과/*.json'
files = glob.glob(PATH)

rows = []
for file in files:
    with open(file,'r',encoding='utf-8-sig') as f:
        data = json.load(f)

# 데이터 추출 -> 'answer','question'만 들고 올거임
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                q = item.get('question')
                a = item.get('answer')
                if q and a:
                    rows.append({'question': q, 'answer': a})

    elif isinstance(data, dict):
        q = data.get('question')
        a = data.get('answer')
        if q and a:
            rows.append({'question': q, 'answer': a})
def clean_question(text):
    text = re.split(r'\n?\d+\)', text)[0]
    text = text.replace('\n', ' ').strip()
    return text

def clean_answer(text):
    text = re.sub(r'^\d+\)\s*','',text)
    return text

df = pd.DataFrame(rows)
df['question'] = df['question'].apply(clean_question)
df['answer'] = df['answer'].apply(clean_answer)

#결측값 삭제
df.isna().sum()

df.to_csv('qa_dataset.csv',index=False,encoding='utf-8-sig')

# 유저 대화내용 인코딩
df['embedding'] = pd.Series([[]] * len(df))
df['embedding'] = df['question'].map(lambda x:list(model.encode(x)))
print(df.head())














