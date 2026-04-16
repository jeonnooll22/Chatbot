from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

print("모델과 데이터셋을 로드하는 중입니다. 잠시만 기다려주세요...")

# 1. 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 2. 초고속 임베딩 파싱 함수 (np.float32 문제 완벽 해결)
def parse_embedding(s):
    if isinstance(s, (list, np.ndarray)):
        return np.array(s, dtype=float)
    
    if isinstance(s, str):
        # 'np.float32(' 와 괄호들, 쉼표를 전부 공백으로 치환
        clean_str = s.replace('np.float32(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace(',', ' ')
        # 공백 기준으로 숫자를 쪼개서 다시 Numpy float 배열로 변환
        arr = np.fromstring(clean_str, sep=' ')
        
        # 데이터가 손상되어 768차원이 아니면 에러 방지용 빈 배열 반환
        if len(arr) != 768:
            return np.zeros(768)
        return arr
        
    return np.zeros(768)

# 3. 데이터 로드 및 전처리
df = pd.read_csv('./qa_dataset.csv')
df['embedding'] = df['embedding'].apply(parse_embedding)

# 데이터를 (데이터개수, 768) 형태의 2차원 행렬로 쌓기
embeddings = np.vstack(df['embedding'].values)

print(f"✅ 데이터셋 행렬 변환 완료! 형태: {embeddings.shape}") 

# 4. L2 정규화 (유사도 계산 속도 극대화용, 0 나누기 방지 1e-10 추가)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
dataset_embeddings = embeddings / norms

print("🚀 챗봇 서버 로딩 완료! 웹 서버를 시작합니다.")

@app.route('/')
def index():
    # 첫 화면 렌더링
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # 사용자의 메시지 수신
    data = request.get_json()
    user_input = data.get('message')

    if not user_input:
        return jsonify({'response': '메시지를 입력해주세요.'})

    # 사용자 입력 인코딩 및 L2 정규화
    query_embedding = model.encode(user_input)
    query_norm = np.linalg.norm(query_embedding) + 1e-10
    query_normalized = query_embedding / query_norm

    # 💡 넘파이 행렬 내적을 통한 전체 데이터 코사인 유사도 초고속 계산
    similarities = np.dot(dataset_embeddings, query_normalized)

    # 가장 유사도가 높은 응답 추출
    best_index = np.argmax(similarities)
    answer = df.iloc[best_index]['answer']

    # JSON 형태로 클라이언트에 반환
    return jsonify({'response': str(answer)})

if __name__ == '__main__':
    # 로컬 환경에서 5000번 포트로 서버 실행
    app.run(debug=True, port=5000)
