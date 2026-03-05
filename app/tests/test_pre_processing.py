import pandas as pd
import pandas.testing as pdt
import pytest

from app.src.pre_processing import pre_processing as pp


@pytest.fixture
def df_dados_padrao():
    """Gera um dataframe novo com os dados novos para usar nos testes."""
    return pd.DataFrame(
        {
            "ID": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Idade": [12, 15, 20, 18, 18, 13, 12, 9, 10, 21, 17],
            "Estado": [
                "SP",
                "BH",
                "RJ",
                "AM",
                "SP",
                "SP",
                "SP",
                "BH",
                "AM",
                "RJ",
                "SP",
            ],
            "Gênero": [
                "menino",
                "menino",
                "Feminino",
                "feminino",
                "Masculino",
                "masculino",
                "Menino",
                "Menina",
                "Masculino",
                "Menino",
                "Menina",
            ],
            "INDE": ["7.5", 8.0, 2.0, 3.5, 5.5, 8.0, 6.0, 7.5, 8.0, 9.5, 1.5],
            "IAA": [6.0, 8.0, 2.5, 3.0, 5.0, 5.5, 7.0, 9.0, 2.0, 1.0, 5.0],
            "IEG": [8.0, 2.0, 3.5, 5.0, 2.5, 3.0, 8.0, 9.0, 2.0, 1.0, 5.0],
            "IPS": [3.0, 7.0, 9.0, 2.0, 1.0, 5.0, 7.5, 8.0, 2.0, 3.5, 5.5],
            "Ida": [2.5, 3.0, 7.0, 9.0, 2.0, 6.0, 5.0, 2.5, 3.0, 5.0, 5.5],
            "Matem": [2.0, 1.0, 5.0, 7.5, None, 2.0, 7.5, 5.0, 2.5, None, 5.0],
            "portug": [2.5, 3.0, 5.0, 5.5, 7.0, 9.0, None, 9.0, 2.0, 1.0, 5.0],
            "ingles": [None, 6.0, None, None, 9.5, 1.5, 1.0, 5.0, 7.5, None, 2.0],
            "IpV": [2.0, 1.0, 5.0, 7.5, 8.0, 2.0, 2.0, 1.0, 5.0, 7.5, 8.0],
            "IAN": [9.0, 2.0, 6.0, 5.0, 2.5, 3.0, 1.5, 1.0, 5.0, 7.5, 8.0],
        }
    )


def test_func_exclude_columns(df_dados_padrao):
    df_resultado = pp.exclude_columns(df_dados_padrao, ["ID", "Estado"])

    assert "ID" not in df_resultado.columns
    assert "Estado" not in df_resultado.columns

    assert "Idade" in df_resultado.columns


def test_func_padronize_collumns(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)

    assert df_resultado.columns.str.islower().all()

    assert not df_resultado.columns.str.isupper().all()


def test_func_padronize_column_gender(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)
    df_resultado = pp.padronize_column_gender(df_resultado)

    genero_esperado = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]

    assert df_resultado["genero"].tolist() == genero_esperado


def test_func_padronize_names_for_collumns(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)
    df_resultado = pp.padronize_names_for_collumns(df_resultado)

    colunas_ajustadas_esperado = [
        "id",
        "idade",
        "estado",
        "genero",
        "inde",
        "iaa",
        "ieg",
        "ips",
        "ida",
        "mat",
        "por",
        "ing",
        "ipv",
        "ian",
    ]

    assert df_resultado.columns.tolist() == colunas_ajustadas_esperado


def test_func_studied_english(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)
    df_resultado = pp.padronize_names_for_collumns(df_resultado)
    df_resultado = pp.studied_english(df_resultado)

    estudaram_binario_esperado = [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1]

    assert "estudou_ingles" in df_resultado.columns
    assert df_resultado["estudou_ingles"].tolist() == estudaram_binario_esperado


def test_func_aplication_median_for_nan(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)
    df_resultado = pp.padronize_names_for_collumns(df_resultado)

    mediana_esperada_mat = df_resultado["mat"].median()
    mediana_esperada_por = df_resultado["por"].median()

    df_resultado = pp.aplication_median_for_nan(df_resultado, "ing")

    # Verifica se as colunas não podem ter mais um valor NaN
    assert df_resultado["mat"].isna().sum() == 0
    assert df_resultado["por"].isna().sum() == 0

    # Verifica se adicionou o valor correto da mediana da coluna
    assert df_resultado.loc[4, "mat"] == mediana_esperada_mat
    assert df_resultado.loc[6, "por"] == mediana_esperada_por

    # Verifica se ainda está o valor NaN na coluna Ing que pedi para não alterar
    assert pd.isna(df_resultado.loc[0, "ing"])


def test_func_correction_collum_inde(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)
    df_resultado = pp.padronize_names_for_collumns(df_resultado)
    df_resultado = pp.correction_collum_inde(df_resultado)

    assert df_resultado["inde"].dtype == "float64"


def test_func_create_columns_for_discrepancies_in_subjects(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)
    df_resultado = pp.padronize_names_for_collumns(df_resultado)
    df_resultado = pp.studied_english(df_resultado)
    df_resultado = pp.aplication_median_for_nan(df_resultado, "ing")
    df_resultado = pp.create_columns_for_discrepancies_in_subjects(df_resultado)

    assert "diferenca_mat" in df_resultado.columns
    assert "diferenca_por" in df_resultado.columns
    assert "diferenca_ing" in df_resultado.columns

    assert (df_resultado["diferenca_ing"] == 0).any()


def test_func_create_column_for_discrepancie_in_ieg_inde(df_dados_padrao):
    df_resultado = pp.padronize_collumns(df_dados_padrao)
    df_resultado = pp.padronize_names_for_collumns(df_resultado)
    df_resultado = pp.correction_collum_inde(df_resultado)
    df_resultado = pp.create_column_for_discrepancie_in_ieg_inde(df_resultado)

    assert "diferenca_ieg" in df_resultado.columns
    assert "diferenca_inde" in df_resultado.columns
