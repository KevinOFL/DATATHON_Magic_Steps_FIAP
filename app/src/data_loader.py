import pandas as pd
from app.src.pre_processing import pre_processing as pp
from app.config.log_config import logger_data_processing

def get_clean_data(path="app/data/base_de_dados_pede_2024.xlsx"):
    df_2022 = pd.read_excel(path, sheet_name="PEDE2022")
    df_2023 = pd.read_excel(path, sheet_name="PEDE2023")
    df_2024 = pd.read_excel(path, sheet_name="PEDE2024")

    df_filtered_2022 = pp.padronize_collumns(df_2022)
    df_filtered_2023 = pp.padronize_collumns(df_2023)
    df_filtered_2024 = pp.padronize_collumns(df_2024)

    df_filtered_2022 = pp.create_target_variable(
        df_filtered_2022, df_filtered_2023
    )
    df_filtered_2023 = pp.create_target_variable(
        df_filtered_2023, df_filtered_2024
    )

    df_filtered_2022 = pp.exclude_columns(
        df_filtered_2022,
        [
            'ra',
            'fase',
            'turma',
            'nome',
            'ano_nasc',
            'instituicao_de_ensino',
            'pedra_20',
            'pedra_21',
            'pedra_22',
            'cg',
            'cf',
            'ct',
            'no_av',
            'avaliador1',
            'rec_av1',
            'avaliador2',
            'rec_av2',
            'avaliador3',
            'rec_av3',
            'avaliador4',
            'rec_av4',
            'rec_psicologia',
            'indicado',
            'atingiu_pv',
            'fase_ideal',
            'destaque_ieg',
            'destaque_ida',
            'destaque_ipv',
            'defas'
        ]
    )

    df_filtered_2023 = pp.correction_collum_age(df_filtered_2023, 2023)
    df_filtered_2023 = pp.exclude_columns(
        df_filtered_2023,
        [
            'ra',
            'fase',
            'data_de_nasc',
            'pedra_2023',
            'turma',
            'nome_anonimizado',
            'instituicao_de_ensino',
            'pedra_20',
            'pedra_21',
            'pedra_22',
            'pedra_23',
            'inde_22',
            'inde_23',
            'cg',
            'cf',
            'ct',
            'no_av',
            'avaliador1',
            'rec_av1',
            'avaliador2',
            'rec_av2',
            'avaliador3',
            'rec_av3',
            'avaliador4',
            'rec_av4',
            'rec_psicologia',
            'indicado',
            'atingiu_pv',
            'fase_ideal',
            'destaque_ieg',
            'destaque_ida',
            'destaque_ipv',
            'destaque_ipv.1',
            'defasagem',
            'ipp'
        ]
    )

    df_filtered_2024 = pp.exclude_columns(
        df_filtered_2024,
        [
            'ra',
            'fase',
            'pedra_2024',
            'turma',
            'data_de_nasc',
            'nome_anonimizado',
            'instituicao_de_ensino',
            'pedra_20',
            'pedra_21',
            'pedra_22',
            'pedra_23',
            'cg',
            'cf',
            'ct',
            'inde_22',
            'inde_23',
            'no_av',
            'avaliador1',
            'rec_av1',
            'avaliador2',
            'rec_av2',
            'avaliador3',
            'avaliador4',
            'avaliador5',
            'avaliador6',
            'rec_psicologia',
            'indicado',
            'atingiu_pv',
            'fase_ideal',
            'destaque_ieg',
            'destaque_ida',
            'destaque_ipv',
            'escola',
            'ativo/_inativo',
            'ativo/_inativo.1',
            'defasagem',
            'ipp'
        ]
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
    
    return df_filtered_2022, df_filtered_2023, df_filtered_2024

if __name__ == "__main__":
    # Se você rodar python data_loader.py no terminal, ele cai aqui para testar
    logger_data_processing.info("Testando o processamento de dados...")
    df22, df23, df24 = get_clean_data()
    logger_data_processing.info(
        f"Shapes: 2022={df22.shape}, 2023={df23.shape}, 2024={df24.shape}"
    )