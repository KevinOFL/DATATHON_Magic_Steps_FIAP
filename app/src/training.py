import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from app.config.log_config import logger_training
from app.src.data_loader import get_clean_data

df22, df23, df24 = get_clean_data()

y_22 = df22["target"]
y_23 = df23["target"]
X_22 = df22[
    [
        "idade",
        "genero",
        "ano_ingresso",
        "mat",
        "diferenca_mat",
        "por",
        "diferenca_por",
        "ing",
        "diferenca_ing",
        "estudou_ingles",
        "iaa",
        "ieg",
        "diferenca_ieg",
        "ips",
        "ida",
        "ipv",
        "ian",
        "inde",
        "diferenca_inde",
    ]
]
X_23 = df23[
    [
        "idade",
        "genero",
        "ano_ingresso",
        "mat",
        "diferenca_mat",
        "por",
        "diferenca_por",
        "ing",
        "diferenca_ing",
        "estudou_ingles",
        "iaa",
        "ieg",
        "diferenca_ieg",
        "ips",
        "ida",
        "ipv",
        "ian",
        "inde",
        "diferenca_inde",
    ]
]


class training_models:
    mlflow.set_experiment("Previsao_Evasao_Escolar_ONG_Passos_Magicos")

    @staticmethod
    def train_model_random_forest(X_train, y_train, X_test, y_test):
        tags = {
            "model_type": "Random Forest Classifier",
            "developer": "Kevin da Silva Nunes Amaral",
        }

        with mlflow.start_run(run_name="RF_Adivinhacao_de_evasao"):
            for i in range(len(tags)):
                mlflow.set_tag(list(tags.keys())[i], list(tags.values())[i])

            logger_training.info("Iniciando treinamento e rastreamento com mlflow")

            params = {
                "n_estimators": randint(50, 1000),
                "max_depth": randint(5, 45),
                "max_samples": [None, 0.25, 0.5, 0.75, 1.0],
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 15),
                "max_features": ["sqrt", "log2"],
            }

            modelo_rf = RandomForestClassifier(
                random_state=42, class_weight="balanced", n_jobs=-1
            )

            random_search = RandomizedSearchCV(
                estimator=modelo_rf,
                param_distributions=params,
                n_iter=1000,
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1,
            )

            logger_training.info("Iniciando o treinamento do modelo Random Forest...")
            random_search.fit(X_train, y_train)
            logger_training.info("Modelo treinado com sucesso!")

            best_param_rf = random_search.best_params_
            best_model_rf = random_search.best_estimator_
            logger_training.info(
                f"Melhores hiperparâmetros encontrados: {best_param_rf}"
            )
            logger_training.info(
                "Capturando e salvando os melhores paramêtros no mlflow..."
            )
            mlflow.log_params(best_param_rf)
            mlflow.log_param("Qtd_linhas_treino", X_train.shape[0])
            mlflow.log_param("Qtd_colunas", X_train.shape[1])

            logger_training.info(
                "Iniciando a avaliação do modelo no conjunto de teste..."
            )
            previsions = best_model_rf.predict(X_test)
            logger_training.info("Avaliação concluída. Resultados:")

            risc_probabilites = best_model_rf.predict_proba(X_test)[:, 1]
            logger_training.info(f"Probabilidades de risco: {risc_probabilites}")

            logger_training.info("\n" + "=" * 50 + "\n")
            logger_training.info("Resultados do modelo:")

            acuracy = (previsions == y_test).mean()
            logger_training.info(f"Acurácia: {acuracy:.4f}")

            f1 = f1_score(y_test, previsions)
            logger_training.info(f"F1 Score: {f1:.4f}")

            logger_training.info("Relatório de classificação:")
            logger_training.info("\n" + f"{classification_report(y_test, previsions)}")

            fig_matriz, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, previsions, ax=ax, cmap="Blues"
            )
            plt.title("Matriz de Confusão")
            mlflow.log_figure(fig_matriz, "graficos/matriz_confusao_RF.png")
            plt.close()

            fig_features, ax = plt.subplots(figsize=(10, 6))
            importancias = best_model_rf.feature_importances_
            sns.barplot(x=importancias, y=X_train.columns, ax=ax, palette="viridis")
            plt.title("Importância das Variáveis para a Evasão")
            plt.xlabel("Peso na Decisão")
            mlflow.log_figure(fig_features, "graficos/feature_importance_RF.png")
            plt.close()

            mlflow.log_metric("acuracia_teste", acuracy)
            mlflow.log_metric("f1_score_teste", f1)
            mlflow.sklearn.log_model(
                sk_model=best_model_rf,
                artifact_path="modelo_rf",
                registered_model_name="Best_Model_RF",
            )

    @staticmethod
    def train_model_random_forest_with_regua(
        X_train, y_train, X_test, y_test, regua_corte: float
    ):
        tags = {
            "model_type": "Random Forest Classifier",
            "developer": "Kevin da Silva Nunes Amaral",
        }

        with mlflow.start_run(run_name="RF_Adivinhacao_de_evasao_com_regua_de_corte"):
            for i in range(len(tags)):
                mlflow.set_tag(list(tags.keys())[i], list(tags.values())[i])

            logger_training.info("Iniciando treinamento e rastreamento com mlflow")
            params = {
                "n_estimators": randint(50, 1000),
                "max_depth": randint(5, 45),
                "max_samples": [None, 0.25, 0.5, 0.75, 1.0],
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 15),
                "max_features": ["sqrt", "log2"],
            }

            modelo_rf = RandomForestClassifier(
                random_state=42, class_weight="balanced", n_jobs=-1
            )

            random_search = RandomizedSearchCV(
                estimator=modelo_rf,
                param_distributions=params,
                n_iter=1000,
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1,
            )

            logger_training.info("Iniciando o treinamento do modelo Random Forest...")
            random_search.fit(X_train, y_train)
            logger_training.info("Modelo treinado com sucesso!")

            best_param_rf = random_search.best_params_
            best_model_rf = random_search.best_estimator_
            logger_training.info(
                f"Melhores hiperparâmetros encontrados: {random_search.best_params_}"
            )
            logger_training.info(
                "Capturando e salvando os melhores paramêtros no mlflow..."
            )

            mlflow.log_params(best_param_rf)
            mlflow.log_param("Qtd_linhas_treino", X_train.shape[0])
            mlflow.log_param("Qtd_colunas", X_train.shape[1])

            risc_probabilites = best_model_rf.predict_proba(X_test)[:, 1]
            logger_training.info(f"Probabilidades de risco: {risc_probabilites}")

            logger_training.info("TESTE DE RÉGUA DE RISCO")
            # Se o risco do aluno for maior ou igual à régua, ele vira 1 (Evadiu). Senão, 0.
            previsoes_ajustadas = (risc_probabilites >= regua_corte).astype(int)

            acuracia = accuracy_score(y_test, previsoes_ajustadas)
            recall = recall_score(y_test, previsoes_ajustadas)
            precision = precision_score(y_test, previsoes_ajustadas)
            matriz = confusion_matrix(y_test, previsoes_ajustadas)
            f1 = f1_score(y_test, previsoes_ajustadas)

            # Extraindo os números reais da Matriz de Confusão para ficar fácil de entender
            verdadeiros_positivos = matriz[1][
                1
            ]  # Iam sair e o modelo APITOU (Sucesso!)
            falsos_negativos = matriz[1][
                0
            ]  # Iam sair e o modelo FICOU QUIETO (Pior erro)
            falsos_positivos = matriz[0][1]  # Iam ficar, mas o modelo deu FALSO ALARME

            logger_training.info(f"==================================================")
            logger_training.info(f"SE A RÉGUA FOR {regua_corte * 100}% DE RISCO:")
            logger_training.info(f"==================================================")
            logger_training.info(f"- ACURACIA: {acuracia:.4f}")
            logger_training.info(f"- RECALL (Captura): {recall * 100:.1f}%")
            logger_training.info(f"- F1 Score: {f1:.4f}")
            logger_training.info(
                f"- PRECISION (Acerto do Alarme): {precision * 100:.1f}%"
            )
            logger_training.info(
                f"- Alunos Salvos (Apito Correto): {verdadeiros_positivos}"
            )
            logger_training.info(
                f"- Alunos que Escaparam (Ponto Cego): {falsos_negativos}"
            )
            logger_training.info(
                f"- Falsos Alarmes (Apito Errado): {falsos_positivos}\n"
            )

            mlflow.log_param("threshold_regua", regua_corte)
            mlflow.log_metric("acuracia_teste", acuracia)
            mlflow.log_metric("recall_evasao", recall)
            mlflow.log_metric("precision_evasao", precision)
            mlflow.log_metric("f1_score_teste", f1)

            fig_matriz, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, previsoes_ajustadas, ax=ax, cmap="Blues"
            )
            plt.title("Matriz de Confusão")
            mlflow.log_figure(
                fig_matriz, "graficos/matriz_confusao_RF_com_regua_de_corte.png"
            )
            plt.close()

            fig_features, ax = plt.subplots(figsize=(10, 6))
            importancias = best_model_rf.feature_importances_
            sns.barplot(x=importancias, y=X_train.columns, ax=ax, palette="viridis")
            plt.title("Importância das Variáveis para a Evasão")
            plt.xlabel("Peso na Decisão")
            mlflow.log_figure(
                fig_features, "graficos/feature_importance_RF_com_regua_de_corte.png"
            )
            plt.close()

            regua_corte_att = str(regua_corte).replace(".", "_")
            mlflow.sklearn.log_model(
                sk_model=best_model_rf,
                artifact_path="modelo_rf_com_regua_de_corte",
                registered_model_name=f"Best_Model_RF_regua_de_corte_{regua_corte_att}",
            )

    @staticmethod
    def train_model_xgboost(X_train, y_train, X_test, y_test):
        tags = {
            "model_type": "XGBoost Classifier",
            "developer": "Kevin da Silva Nunes Amaral",
        }
        with mlflow.start_run(run_name="XGBoost_Adivinhacao_de_evasao"):
            for i in range(len(tags)):
                mlflow.set_tag(list(tags.keys())[i], list(tags.values())[i])

            logger_training.info("Iniciando treinamento e rastreamento com mlflow")
            param = {
                "n_estimators": randint(50, 1000),
                "max_depth": randint(5, 45),
                "learning_rate": uniform(0.01, 0.39),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.5, 0.5),
                "gamma": uniform(0, 5),
            }

            peso_balanceamento = 3.1

            modelo_xgb = XGBClassifier(
                random_state=42,
                scale_pos_weight=peso_balanceamento,
                eval_metric="logloss",
                n_jobs=-1,
            )

            random_search_xgb = RandomizedSearchCV(
                estimator=modelo_xgb,
                param_distributions=param,
                n_iter=5000,
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1,
            )

            logger_training.info("Iniciando o treinamento do modelo XGBoost...")
            random_search_xgb.fit(X_train, y_train)
            logger_training.info("Modelo treinado com sucesso!")

            best_param_xgb = random_search_xgb.best_params_
            best_model_xgb = random_search_xgb.best_estimator_
            logger_training.info(
                f"Melhores hiperparâmetros encontrados: {best_param_xgb}"
            )
            logger_training.info(
                "Capturando e salvando os melhores paramêtros no mlflow..."
            )
            mlflow.log_params(best_param_xgb)
            mlflow.log_param("Qtd_linhas_treino", X_train.shape[0])
            mlflow.log_param("Qtd_colunas", X_train.shape[1])

            logger_training.info(
                "Iniciando a avaliação do modelo no conjunto de teste..."
            )
            previsions = best_model_xgb.predict(X_test)
            logger_training.info("Avaliação concluída. Resultados:")

            risc_probabilites = best_model_xgb.predict_proba(X_test)[:, 1]
            logger_training.info(f"Probabilidades de risco: {risc_probabilites}")

            logger_training.info("\n" + "=" * 50 + "\n")
            logger_training.info("Resultados do modelo:")

            acuracy = (previsions == y_test).mean()
            logger_training.info(f"Acurácia: {acuracy:.4f}")

            f1 = f1_score(y_test, previsions)
            logger_training.info(f"F1 Score: {f1:.4f}")

            logger_training.info("Relatório de classificação:")
            logger_training.info("\n" + f"{classification_report(y_test, previsions)}")

            fig_matriz, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, previsions, ax=ax, cmap="Blues"
            )
            plt.title("Matriz de Confusão")
            mlflow.log_figure(fig_matriz, "graficos/matriz_confusao_XGBoost.png")
            plt.close()

            fig_features, ax = plt.subplots(figsize=(10, 6))
            importancias = best_model_xgb.feature_importances_
            sns.barplot(x=importancias, y=X_train.columns, ax=ax, palette="viridis")
            plt.title("Importância das Variáveis para a Evasão")
            plt.xlabel("Peso na Decisão")
            mlflow.log_figure(fig_features, "graficos/feature_importance_XGBoost.png")
            plt.close()

            mlflow.log_metric("acuracia_teste", acuracy)
            mlflow.log_metric("f1_score_teste", f1)
            mlflow.sklearn.log_model(
                sk_model=best_model_xgb,
                artifact_path="modelo_xgboost",
                registered_model_name="Best_Model_XGBoost",
            )

    @staticmethod
    def train_model_xgboost_with_regua(
        X_train, y_train, X_test, y_test, regua_corte: float
    ):
        tags = {
            "model_type": "XGBoost_Classifier",
            "developer": "Kevin da Silva Nunes Amaral",
        }

        with mlflow.start_run(run_name="XGBoost_Adivinhacao_de_evasao"):
            for i in range(len(tags)):
                mlflow.set_tag(list(tags.keys())[i], list(tags.values())[i])

            logger_training.info("Iniciando treinamento e rastreamento com mlflow")
            params = {
                "n_estimators": randint(50, 750),
                "max_depth": randint(5, 40),
                "learning_rate": uniform(0.01, 0.29),
                "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.5, 0.5),
                "gamma": uniform(0, 5),
            }

            peso_balanceamento = 3.1

            modelo_xgb = XGBClassifier(
                random_state=42,
                scale_pos_weight=peso_balanceamento,
                eval_metric="logloss",
                n_jobs=-1,
            )

            random_search = RandomizedSearchCV(
                estimator=modelo_xgb,
                param_distributions=params,
                n_iter=5000,
                cv=5,
                verbose=1,
                random_state=42,
                n_jobs=-1,
            )

            logger_training.info("Iniciando o treinamento do modelo XGBoost...")
            random_search.fit(X_train, y_train)
            logger_training.info("Modelo treinado com sucesso!")

            best_param_xgb = random_search.best_params_
            best_model_xgb = random_search.best_estimator_
            logger_training.info(
                f"Melhores hiperparâmetros encontrados: {random_search.best_params_}"
            )
            logger_training.info(
                "Capturando e salvando os melhores paramêtros no mlflow..."
            )

            mlflow.log_params(best_param_xgb)
            mlflow.log_param("Qtd_linhas_treino", X_train.shape[0])
            mlflow.log_param("Qtd_colunas", X_train.shape[1])

            risc_probabilites = best_model_xgb.predict_proba(X_test)[:, 1]
            logger_training.info(f"Probabilidades de risco: {risc_probabilites}")

            logger_training.info("TESTE DE RÉGUA DE RISCO")
            # Se o risco do aluno for maior ou igual à régua, ele vira 1 (Evadiu). Senão, 0.
            previsoes_ajustadas = (risc_probabilites >= regua_corte).astype(int)

            acuracia = accuracy_score(y_test, previsoes_ajustadas)
            recall = recall_score(y_test, previsoes_ajustadas)
            precision = precision_score(y_test, previsoes_ajustadas)
            matriz = confusion_matrix(y_test, previsoes_ajustadas)
            f1 = f1_score(y_test, previsoes_ajustadas)

            # Extraindo os números reais da Matriz de Confusão para ficar fácil de entender
            verdadeiros_positivos = matriz[1][
                1
            ]  # Iam sair e o modelo APITOU (Sucesso!)
            falsos_negativos = matriz[1][
                0
            ]  # Iam sair e o modelo FICOU QUIETO (Pior erro)
            falsos_positivos = matriz[0][1]  # Iam ficar, mas o modelo deu FALSO ALARME

            logger_training.info(f"==================================================")
            logger_training.info(f"SE A RÉGUA FOR {regua_corte * 100}% DE RISCO:")
            logger_training.info(f"==================================================")
            logger_training.info(f"- ACURACIA: {acuracia:.4f}")
            logger_training.info(f"- RECALL (Captura): {recall * 100:.1f}%")
            logger_training.info(f"- F1 Score: {f1:.4f}")
            logger_training.info(
                f"- PRECISION (Acerto do Alarme): {precision * 100:.1f}%"
            )
            logger_training.info(
                f"- Alunos Salvos (Apito Correto): {verdadeiros_positivos}"
            )
            logger_training.info(
                f"- Alunos que Escaparam (Ponto Cego): {falsos_negativos}"
            )
            logger_training.info(
                f"- Falsos Alarmes (Apito Errado): {falsos_positivos}\n"
            )

            mlflow.log_param("threshold_regua", regua_corte)
            mlflow.log_metric("acuracia_teste", acuracia)
            mlflow.log_metric("recall_evasao", recall)
            mlflow.log_metric("precision_evasao", precision)
            mlflow.log_metric("f1_score_teste", f1)

            fig_matriz, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, previsoes_ajustadas, ax=ax, cmap="Blues"
            )
            plt.title("Matriz de Confusão")
            mlflow.log_figure(
                fig_matriz, "graficos/matriz_confusao_XGBoost_com_regua_de_corte.png"
            )
            plt.close()

            fig_features, ax = plt.subplots(figsize=(10, 6))
            importancias = best_model_xgb.feature_importances_
            sns.barplot(x=importancias, y=X_train.columns, ax=ax, palette="viridis")
            plt.title("Importância das Variáveis para a Evasão")
            plt.xlabel("Peso na Decisão")
            mlflow.log_figure(
                fig_features,
                "graficos/feature_importance_XGBoost_com_regua_de_corte.png",
            )
            plt.close()

            regua_corte_att = str(regua_corte).replace(".", "_")
            mlflow.sklearn.log_model(
                sk_model=best_model_xgb,
                artifact_path="modelo_xgboost_com_regua_de_corte",
                registered_model_name=f"Best_Model_XGBoost_regua_de_corte_{regua_corte_att}",
            )


# training_models.train_model_xgboost(X_22, y_22, X_23, y_23)
# training_models.train_model_random_forest(X_22, y_22, X_23, y_23)
# training_models.train_model_random_forest_with_regua(X_22, y_22, X_23, y_23, 0.30)
training_models.train_model_xgboost_with_regua(X_22, y_22, X_23, y_23, 0.34)
