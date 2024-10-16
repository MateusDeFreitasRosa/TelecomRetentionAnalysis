import os
import pandas as pd
import joblib
import json
from sklearn.preprocessing import OneHotEncoder

# Função para carregar o modelo e o encoder treinado no SageMaker
def model_fn(model_dir):
    """Carregar o modelo e o encoder treinado"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    encoder = joblib.load(os.path.join(model_dir, "encoder.joblib"))
    return model, encoder

# Função para pré-processar os dados de entrada
def input_fn(request_body, request_content_type='application/json'):
    """Pré-processar os dados de entrada"""
    if request_content_type == 'application/json':
        data = pd.read_json(request_body, orient='records')
    elif request_content_type == 'text/csv':
        from io import StringIO
        data = pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Content type {request_content_type} não suportado")
    
    # (O restante do código permanece o mesmo)
    # Excluir a coluna 'customerID' (se presente)
    data = data.drop(columns=['customerID'], errors='ignore')

    # Converter a coluna 'TotalCharges' para numérico, tratando valores inválidos
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Preencher valores faltantes na coluna 'TotalCharges' com a mediana
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

    # As colunas categóricas que precisam ser codificadas
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']

    # Retorna os dados pré-processados e as colunas categóricas para codificação
    return data, categorical_columns


# Função para fazer a predição com o modelo
def predict_fn(input_data, model_and_encoder):
    """Executa a inferência usando o modelo e o encoder"""
    model, encoder = model_and_encoder

    # Aplicar o OneHotEncoder nas colunas categóricas
    data, categorical_columns = input_data

    # Codificar as colunas categóricas
    encoded_features = encoder.transform(data[categorical_columns])

    # Criar um DataFrame com as colunas categóricas codificadas
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names(categorical_columns))

    # Remover as colunas categóricas originais e adicionar as codificadas
    data = data.drop(columns=categorical_columns)
    data.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    data = pd.concat([data, encoded_df], axis=1)

    # Fazer a predição com o modelo
    predictions = model.predict(data)
    return predictions

# Função para retornar os resultados no formato adequado (JSON)
def output_fn(prediction, accept='application/json'):
    """Retorna o resultado da inferência no formato JSON"""
    if accept == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    else:
        raise ValueError(f"Tipo de saída {accept} não suportado")
