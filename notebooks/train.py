import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

# Função para carregar o modelo treinado no SageMaker
def model_fn(model_dir):
    """Carregar o modelo treinado"""
    return joblib.load(os.path.join(model_dir, "model.joblib"))

if __name__ == '__main__':
    # Ler os argumentos fornecidos pelo SageMaker
    parser = argparse.ArgumentParser()

    # Defina os argumentos esperados
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=100)
    
    # Defina diretórios com valores padrão se a variável de ambiente não estiver definida
    parser.add_argument('--model-dir', type=str, default=os.getenv('SM_MODEL_DIR', 'model'))
    parser.add_argument('--output-dir', type=str, default=os.getenv('SM_OUTPUT_DIR', 'output'))
    parser.add_argument('--train', type=str, default=os.getenv('SM_CHANNEL_TRAIN', 'data/train'))

    args = parser.parse_args()

    args = parser.parse_args()

    # Carregar os dados de treinamento do S3
    dataset_path = os.path.join(args.train, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(dataset_path)

    # Pré-processamento básico
    # Converter a variável de resposta 'Churn' para 0 e 1
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Separar as features (X) da variável alvo (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Separar os dados de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instanciar e treinar o modelo XGBoost
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc}")

    # Salvar o modelo no diretório do SageMaker
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
