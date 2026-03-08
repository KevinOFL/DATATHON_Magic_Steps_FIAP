import pandas as pd

from app.config.log_config import logger_data_processing
from app.src.pre_processing import pre_processing as pp


def get_clean_data(path="app/data/base_de_dados_pede_2024.xlsx"):
    """
    Executa o pipeline completo de pré-processamento para a base histórica da ONG.

    Lê os dados brutos das planilhas referentes aos anos de 2022, 2023 e 2024, e 
    aplica uma sequência de transformações para limpeza e padronização. Isso inclui 
    a criação da variável alvo de evasão, exclusão de colunas não preditivas, 
    correção de tipagens (idade, inde), padronização de nomenclatura de colunas e 
    gêneros, criação de variáveis dummy (estudou inglês), imputação de valores 
    nulos via mediana e engenharia de features (colunas de discrepâncias em notas e 
    indicadores).

    parametros:
    - path (str): Caminho absoluto ou relativo para o arquivo Excel contendo as bases anuais. (Opcional, com valor padrão).

    retorno:
    - tuple: Retorna uma tupla contendo três DataFrames (pd.DataFrame) na seguinte ordem: dados_2022, dados_2023, dados_2024, todos limpos e prontos para modelagem.
    """
    df_2022 = pd.read_excel(path, sheet_name="PEDE2022")
    df_2023 = pd.read_excel(path, sheet_name="PEDE2023")
    df_2024 = pd.read_excel(path, sheet_name="PEDE2024")

    df_filtered_2022 = pp.padronize_collumns(df_2022)
    df_filtered_2023 = pp.padronize_collumns(df_2023)
    df_filtered_2024 = pp.padronize_collumns(df_2024)

    df_filtered_2022 = pp.create_target_variable(df_filtered_2022, df_filtered_2023)
    df_filtered_2023 = pp.create_target_variable(df_filtered_2023, df_filtered_2024)

    df_filtered_2022 = pp.exclude_columns(
        df_filtered_2022,
        [
            "ra",
            "fase",
            "turma",
            "nome",
            "ano_nasc",
            "instituicao_de_ensino",
            "pedra_20",
            "pedra_21",
            "pedra_22",
            "cg",
            "cf",
            "ct",
            "no_av",
            "avaliador1",
            "rec_av1",
            "avaliador2",
            "rec_av2",
            "avaliador3",
            "rec_av3",
            "avaliador4",
            "rec_av4",
            "rec_psicologia",
            "indicado",
            "atingiu_pv",
            "fase_ideal",
            "destaque_ieg",
            "destaque_ida",
            "destaque_ipv",
            "defas",
        ],
    )

    df_filtered_2023 = pp.correction_collum_age(df_filtered_2023, 2023)
    df_filtered_2023 = pp.exclude_columns(
        df_filtered_2023,
        [
            "ra",
            "fase",
            "data_de_nasc",
            "pedra_2023",
            "turma",
            "nome_anonimizado",
            "instituicao_de_ensino",
            "pedra_20",
            "pedra_21",
            "pedra_22",
            "pedra_23",
            "inde_22",
            "inde_23",
            "cg",
            "cf",
            "ct",
            "no_av",
            "avaliador1",
            "rec_av1",
            "avaliador2",
            "rec_av2",
            "avaliador3",
            "rec_av3",
            "avaliador4",
            "rec_av4",
            "rec_psicologia",
            "indicado",
            "atingiu_pv",
            "fase_ideal",
            "destaque_ieg",
            "destaque_ida",
            "destaque_ipv",
            "destaque_ipv.1",
            "defasagem",
            "ipp",
        ],
    )

    df_filtered_2024 = pp.exclude_columns(
        df_filtered_2024,
        [
            "ra",
            "fase",
            "pedra_2024",
            "turma",
            "data_de_nasc",
            "nome_anonimizado",
            "instituicao_de_ensino",
            "pedra_20",
            "pedra_21",
            "pedra_22",
            "pedra_23",
            "cg",
            "cf",
            "ct",
            "inde_22",
            "inde_23",
            "no_av",
            "avaliador1",
            "rec_av1",
            "avaliador2",
            "rec_av2",
            "avaliador3",
            "avaliador4",
            "avaliador5",
            "avaliador6",
            "rec_psicologia",
            "indicado",
            "atingiu_pv",
            "fase_ideal",
            "destaque_ieg",
            "destaque_ida",
            "destaque_ipv",
            "escola",
            "ativo/_inativo",
            "ativo/_inativo.1",
            "defasagem",
            "ipp",
        ],
    )

    df_filtered_2022 = pp.padronize_names_for_collumns(df_filtered_2022)
    df_filtered_2023 = pp.padronize_names_for_collumns(df_filtered_2023)
    df_filtered_2024 = pp.padronize_names_for_collumns(df_filtered_2024)

    df_filtered_2022 = pp.padronize_column_gender(df_filtered_2022)
    df_filtered_2023 = pp.padronize_column_gender(df_filtered_2023)
    df_filtered_2024 = pp.padronize_column_gender(df_filtered_2024)

    df_filtered_2022 = pp.studied_english(df_filtered_2022)
    df_filtered_2023 = pp.studied_english(df_filtered_2023)
    df_filtered_2024 = pp.studied_english(df_filtered_2024)

    df_filtered_2022 = pp.aplication_median_for_nan(df_filtered_2022, "ing")
    df_filtered_2023 = pp.aplication_median_for_nan(df_filtered_2023, "ing")

    df_filtered_2024 = pp.correction_collum_inde(df_filtered_2024)
    df_filtered_2024 = pp.aplication_median_for_nan(df_filtered_2024, "ing")

    df_filtered_2022 = pp.create_columns_for_discrepancies_in_subjects(df_filtered_2022)
    df_filtered_2023 = pp.create_columns_for_discrepancies_in_subjects(df_filtered_2023)
    df_filtered_2024 = pp.create_columns_for_discrepancies_in_subjects(df_filtered_2024)

    df_filtered_2022 = pp.create_column_for_discrepancie_in_ieg_inde(df_filtered_2022)
    df_filtered_2023 = pp.create_column_for_discrepancie_in_ieg_inde(df_filtered_2023)
    df_filtered_2024 = pp.create_column_for_discrepancie_in_ieg_inde(df_filtered_2024)

    return df_filtered_2022, df_filtered_2023, df_filtered_2024


def loading_data_prediction(archive):
    """
    Executa o pipeline de pré-processamento em novos dados recebidos para inferência.

    Recebe um DataFrame bruto (geralmente recém-carregado via upload na API) e 
    aplica estritamente as mesmas regras de transformação utilizadas no treinamento 
    do modelo. Isso garante que os dados entrem no modelo (XGBoost) com a exata 
    mesma estrutura, tipagem e engenharia de features (exclusão de colunas inúteis, 
    padronização, dummies, imputação de mediana e colunas de discrepância).

    parametros:
    - archive (pd.DataFrame): O DataFrame bruto recebido pela requisição de predição.

    retorno:
    - pd.DataFrame: Um novo DataFrame limpo, padronizado e com as features de engenharia de dados aplicadas, pronto para ser submetido à função .predict() do modelo.
    """
    df = pp.padronize_collumns(archive)
    df = pp.exclude_columns(
        df,
        [
            "ra",
            "fase",
            "pedra_2024",
            "turma",
            "data_de_nasc",
            "nome_anonimizado",
            "instituicao_de_ensino",
            "pedra_20",
            "pedra_21",
            "pedra_22",
            "pedra_23",
            "cg",
            "cf",
            "ct",
            "inde_22",
            "no_av",
            "avaliador1",
            "rec_av1",
            "avaliador2",
            "rec_av2",
            "avaliador3",
            "avaliador4",
            "avaliador5",
            "avaliador6",
            "rec_psicologia",
            "indicado",
            "atingiu_pv",
            "fase_ideal",
            "destaque_ieg",
            "destaque_ida",
            "destaque_ipv",
            "escola",
            "ativo/_inativo",
            "ativo/_inativo.1",
            "defasagem",
            "ipp",
        ],
    )
    df = pp.padronize_names_for_collumns(df)
    df = pp.padronize_column_gender(df)
    df = pp.studied_english(df)
    df = pp.aplication_median_for_nan(df, "ing")
    df = pp.correction_collum_inde(df)
    df = pp.aplication_median_for_nan(df, "ing")
    df = pp.create_columns_for_discrepancies_in_subjects(df)
    df = pp.create_column_for_discrepancie_in_ieg_inde(df)

    return df

def loading_data(archive, exclude_collumns: list):
    """
    Executa o pipeline de pré-processamento em novos dados recebidos.

    Recebe um DataFrame bruto (geralmente recém-carregado via upload na API) e 
    aplica estritamente as mesmas regras de transformação utilizadas no treinamento 
    do modelo. Isso garante que os dados entrem no algoritmo com a exata mesma 
    estrutura, tipagem e engenharia de features (exclusão de colunas inúteis 
    recebidas via parâmetro, padronização, dummies, imputação de mediana e 
    colunas de discrepância).

    parametros:
    - archive (pd.DataFrame): O DataFrame bruto recebido pela requisição de predição.
    - exclude_collumns (list): Lista de strings contendo os nomes das colunas que devem ser removidas do DataFrame original.

    retorno:
    - pd.DataFrame: Um novo DataFrame limpo, padronizado e com as features de engenharia de dados aplicadas, pronto para ser submetido à função .predict() do modelo.
    """
    df = pp.padronize_collumns(archive)
    df = pp.exclude_columns(
        df,
        exclude_collumns,
    )
    df = pp.padronize_names_for_collumns(df)
    df = pp.padronize_column_gender(df)
    df = pp.studied_english(df)
    df = pp.aplication_median_for_nan(df, "ing")
    df = pp.correction_collum_inde(df)
    df = pp.aplication_median_for_nan(df, "ing")
    df = pp.create_columns_for_discrepancies_in_subjects(df)
    df = pp.create_column_for_discrepancie_in_ieg_inde(df)

    return df


if __name__ == "__main__":
    # Se você rodar python data_loader.py no terminal, ele cai aqui para testar
    logger_data_processing.info("Testando o processamento de dados...")
    df22, df23, df24 = get_clean_data()
    logger_data_processing.info(
        f"Shapes: 2022={df22.shape}, 2023={df23.shape}, 2024={df24.shape}"
    )
