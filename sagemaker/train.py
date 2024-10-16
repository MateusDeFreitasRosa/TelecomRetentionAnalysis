import argparse
import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
    # Ler os argumentos fornecidos pelo SageMaker
    parser = argparse.ArgumentParser()

    # Hiperparâmetros para XGBoost
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.1)

    # Diretórios de entrada e saída (para SageMaker)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Caminho do arquivo de treinamento
    train_data_path = os.path.join(args.train, "train.csv")
    df = pd.read_csv(train_data_path)
    
    print('Shape Train: {}'.format(df.shape))

    ########################################################### Pré-processamento ###########################################################

    # Excluir a coluna 'customerID'
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Converter a coluna 'TotalCharges' para numérico e tratar valores faltantes
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Identificar as colunas categóricas
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']

    # Instanciar o OneHotEncoder e codificar as colunas categóricas
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(df[categorical_columns])

    # Usar get_feature_names() ao invés de get_feature_names_out()
    encoded_columns = encoder.get_feature_names(categorical_columns)

    # Criar um DataFrame com as colunas codificadas
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_columns)

    # Remover as colunas categóricas originais e adicionar as codificadas
    df = df.drop(columns=categorical_columns)
    df = pd.concat([df.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

    ########################################################### Pré-processamento ###########################################################

    # Separar as features (X) da variável alvo (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']


    # Instanciar e treinar o modelo XGBoost
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X, y)


    # Salvar o modelo no diretório do SageMaker
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

    # Salvar o encoder para uso futuro (opcional)
    joblib.dump(encoder, os.path.join(args.model_dir, "encoder.joblib"))
