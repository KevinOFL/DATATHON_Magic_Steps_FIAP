import pandas as pd

class pre_processing:
    def exclude_columns(df, columns_to_exclude: list):
        """
        Exclui colunas específicas de um DataFrame.

        parametros:
        - df (pd.DataFrame): O DataFrame original.
        - columns_to_exclude (list): Lista de nomes de colunas a serem excluídas.

        retorno:
        - pd.DataFrame: Um novo DataFrame sem as colunas especificadas.
        """
        # Cria uma cópia do DataFrame para evitar modificar o original
        df_copy = df.copy()  
        df_copy = df_copy.drop(columns=columns_to_exclude)
        return df_copy
    
    def padronize_collumns(df):
        """
        Padroniza os nomes das colunas de um DataFrame, removendo espaços,
        convertendo para minúsculas e normalizando caracteres acentuados.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.

        retorno:
        - pd.DataFrame: Um novo DataFrame com os nomes das colunas padronizados.
        """
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.strip().str.lower().str.replace(' ', '_')
        df_copy.columns = df_copy.columns.str.normalize(
            'NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        return df_copy
    
    def padronize_names_for_collumns(df):
        """
        Padroniza os nomes das colunas de um DataFrame,
        renomeando colunas específicas de acordo com um mapeamento pré-definido.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.
        
        retorno:
        - pd.DataFrame: Um novo DataFrame com os nomes das colunas padronizados
        """
        
        collumns_mapre_processinging = {
            'idade_22': 'idade', 'inde_22': 'inde', 'inde_2023': 'inde',
            'inde_2024': 'inde', 'matem': 'mat', 'portug': 'por', 'ingles': 'ing',
            'defas': 'defasagem'    
        }
        
        df_copy = df.copy()
        
        for key, value in collumns_mapre_processinging.items():
            for col in df_copy.columns:
                if key in col:
                    new_col_name = str(col).replace(key, value)
                    df_copy.rename(columns={col: new_col_name}, inplace=True)
                    
        return df_copy
    
    import pandas as pd

    def correction_collum_age(df, year):
        """
        Corrige a coluna de idade recalculando-a a partir da data de nascimento.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.
        - year (int): O ano de referência para o cálculo da idade.
        
        retorno:
        - pd.DataFrame: Um novo DataFrame com a coluna de idade corrigida.
        """
        df_copy = df.copy()
        
        datas_nascimento = pd.to_datetime(df_copy['data_de_nasc'], errors='coerce')
        
        anos_nascimento = datas_nascimento.dt.year
        
        df_copy['idade'] = year - anos_nascimento
        
        return df_copy
    
    def padronize_column_gender(df):
        """
        Padroniza a coluna de gênero, convertendo os valores
        para um formato consistente e transforma para binario.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.
        
        retorno:
        - pd.DataFrame: Um novo DataFrame com a coluna de gênero padronizada.
        """
        df_copy = df.copy()
        
        df_copy['genero'] = df_copy['genero'].str.strip().str.lower()
        
        df_copy['genero'] = df_copy['genero'].replace({
            'masculino': 'masculino',
            'menino': 'masculino',
            'feminino': 'feminino',
            'menina': 'feminino',
        })
        
        df_copy['genero'] = df_copy['genero'].map({'masculino': 0, 'feminino': 1})
        
        return df_copy
    
    def studied_english(df):
        """
        Cria uma nova coluna 'estudou_ingles' indicando
        se o aluno estudou inglês ou não.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.
        
        retorno:
        - pd.DataFrame: Um novo DataFrame com a coluna 'estudou_ingles' adicionada.
        """
        
        df_copy = df.copy()
        df_copy['estudou_ingles'] = df_copy['ing'].apply(lambda x: 1 if x > 0 else 0)
        return df_copy
    
    def aplication_median_for_nan(df, colunas_ignorar=None):
        """
        Aplica a mediana para preencher valores NaN em um DataFrame,
        ignorando colunas especificadas.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.
        - colunas_ignorar (list): Lista com os nomes das colunas que
        NÃO devem ser alteradas.
        
        retorno:
        - pd.DataFrame: Um novo DataFrame com os valores NaN preenchidos pela mediana.
        """
        # Se não passar nenhuma coluna para ignorar, cria uma lista vazia
        if colunas_ignorar is None:
            colunas_ignorar = []
            
        df_copy = df.copy()
        
        for column in df_copy.columns:
            # REGRA DE PULO: Se o nome da coluna estiver na lista de ignoradas,
            # pula para a próxima
            if column in colunas_ignorar:
                continue
                
            # Verifica se a coluna é numérica (int ou float)
            if df_copy[column].dtype in ['float64', 'int64']:
                median_value = df_copy[column].median()
                
                # Atualiza os valores NaN com a mediana calculada
                df_copy[column] = df_copy[column].fillna(median_value)
                
        return df_copy
    
    def correction_collum_inde(df):
        """
        Corrige a coluna 'inde' convertendo seus valores para numéricos,
        tratando erros de conversão.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.
        
        retorno:
        - pd.DataFrame: Um novo DataFrame com a coluna 'inde' corrigida.
        """
        df_copy = df.copy()
        inde = pd.to_numeric(df_copy['inde'], errors='coerce')
        df_copy['inde'] = inde
        return df_copy
    
    def create_target_variable(df_base, df_atual):
        """
        Cria a variável alvo 'target' com base na coluna 'defasagem',
        onde o valor 1 indica que evadiu e 0 caso permaneceu.
        
        parametros:
        - df (pd.DataFrame): O DataFrame original.
        
        retorno:
        - pd.DataFrame: Um novo DataFrame com a coluna 'target' adicionada.
        """
        df_copy_base = df_base.copy()
        df_copy_atual = df_atual.copy()
        
        ras_base = set(df_copy_base['ra'])
        ras_atual = set(df_copy_atual['ra'])
        
        ras_evadidados = ras_base - ras_atual
        df_copy_base['target'] = df_copy_base['ra'].isin(ras_evadidados).astype(int)
        
        return df_copy_base