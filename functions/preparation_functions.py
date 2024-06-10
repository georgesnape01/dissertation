######################################################################################################################## duplicate functions

def find_duplicates_features(df):
    dupe_cols = np.transpose(df).duplicated()
    dupe_cols = df.columns[dupe_cols].tolist()
    dupes_of = {}
    for col_name in dupe_cols:
        col_values = df[col_name]
        dupes = [other_col for other_col in df.columns if (other_col != col_name) and df[other_col].equals(col_values)]
        dupes_of[col_name] = dupes
        print(f'{col_name} is a duplicate of: {dupes}')

def drop_duplicate_features(df):
    dupe_cols = np.transpose(df).duplicated()
    dupe_cols = df.columns[dupe_cols].tolist()
    df = df.drop(columns = dupe_cols)
    return df

######################################################################################################################## outlier functions

def column_contents(df):
    pot_cols = []
    for col in df.columns:
        if not df[col].isin([0, 1, pd.NA]).all():
            pot_cols.append(col)
    if len(pot_cols) == 0:
        print(f'Columns only contain 0, 1 and <NA>')
    else:
        print(f'Columns contining more than 0, 1 and <NA>: {pot_cols}')
    return pot_cols

def mad(df):
    median = df.median()
    deviations = np.abs(df - median)
    mad_val = deviations.median() # MAD
    return mad_val

def modified_zscore(df, threshold = 3.5):
    median = df.median()
    mad_val = mad(df) # MAD
    modified_zscores = 0.6745 * (df - median) / mad_val # modified Z-score
    return np.abs(modified_zscores) > threshold

def find_outlier_cols(df, threshold = 3.5): # counts also

    outlier_df = modified_zscore(df)
    outlier_cols = set()
    for col in outlier_df.columns:
        outlier_list = []
        for i, val in outlier_df[col].items():
            if pd.notna(val) and val:
                outlier_list.append(df.iloc[i][col])
        if outlier_list:
            outlier_cols.add(col)
            print(f'{col} outlier count: {len(outlier_list)}')

    # return outlier_df
    outlier_cols = list(outlier_cols)
    outlier_df = df[outlier_cols]
    return outlier_df

def plot_outlier_cols(outlier_df, title, legend=True):
    outlier_df.plot(figsize = (5, 3))
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Total Score')
    if legend:
        plt.legend(loc='upper right')
    else:
        plt.gca().legend().set_visible(False)
    plt.show()

def plot_datetime_features(df):
    plt.figure(figsize=(10, 4))
    for i, col in enumerate(datetime_cols):
        plt.subplot(1, 2, i + 1)
        df[col].hist(bins = 100, color = 'grey', edgecolor = 'white')
        plt.title(col)
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_outlier_cols2(outlier_df, title, color):

    plot_cols = 4
    plot_rows = len(outlier_df.columns)//4 + 1
    plt.figure(figsize = (plot_cols*3, plot_rows*3))

    for i, col in enumerate(outlier_df.columns, start = 1):
        plt.subplot(plot_rows, 4, i)
        outlier_df[col].plot(color=color)
        plt.title(col)
        plt.xlabel('Index')
        plt.ylabel('Total Score')

    plt.suptitle(title, y = 1, fontsize = 24)
    plt.tight_layout()
    plt.show()

def replace_ques_outliers(df): # shortens search using quartiles

    item_cols = [col for col in df.columns if 'Item' in col]
    item_df = df[item_cols]

    total_cols = [col for col in df.columns if 'Total' in col]
    total_df = df[total_cols]

    # replace negative totals with nan
    for col in total_df:
        df.loc[df[col] < 0, col] = np.nan # replace in df
        total_df.loc[total_df[col] < 0, col] = np.nan # replace in total_df

    # find col contents
    def create_pot_cols(df):
        pot_cols = []
        for col in df.columns:
            if not df[col].isin([0, 1, pd.NA]).all():
                pot_cols.append(col)
        return pot_cols

    # detect whether cols are just 0,1,na
    item_pot_cols = create_pot_cols(item_df)
    total_pot_cols = create_pot_cols(total_df)

    # potential columns containing outliers (shortening the search, only these can contain outliers)
    total_pot_df = total_df[total_pot_cols]
    item_pot_df = item_df[item_pot_cols]

    # outlier counter and df function
    def create_outlier_df(df, threshold = 3.5):

        outlier_df = modified_zscore(df)
        outlier_cols = set()
        for col in outlier_df.columns:
            outlier_list = []
            for i, val in outlier_df[col].items():
                if pd.notna(val) and val:
                    outlier_list.append(df.iloc[i][col])
            if outlier_list:
                outlier_cols.add(col)

        # return outlier_df
        outlier_cols = list(outlier_cols)
        outlier_df = df[outlier_cols]
        return outlier_df

    # find outliers in variables
    total_outlier_df = create_outlier_df(total_pot_df)
    item_outlier_df = create_outlier_df(item_pot_df)

    # replace outliers with nan
    for col in total_outlier_df:
        df.loc[df[col] > 200, col] = np.nan
        total_outlier_df.loc[total_outlier_df[col] > 200, col] = np.nan

    for col in item_outlier_df:
        df.loc[df[col] > 50, col] = np.nan
        item_outlier_df.loc[item_outlier_df[col] > 50, col] = np.nan

    df.loc[df['Total14'] > 37, 'Total14'] = np.nan
    df.loc[df['Item70'] > 3, 'Item70'] = np.nan
    df.loc[df['Item132'] > 5, 'Item132'] = np.nan

    return df

def replace_outliers(df):
    df.loc[df['AgeAtReferralRequest'] == 0, 'AgeAtReferralRequest'] = np.nan
    df = replace_ques_outliers(df)
    return df

######################################################################################################################## constant/quasi constant functions

def drop_const_features(df):
    const_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns = const_cols)
    return df

def plot_datetime_features(df):
    plt.figure(figsize = (10,4))
    for i, col in enumerate(datetime_cols, 1):
        plt.subplot(1, len(datetime_cols), i)
        plt.hist(df[col], bins = 20, color = 'grey', edgecolor = 'white')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(col)
    plt.tight_layout()
    plt.show()

def quasi_percentage(df):

    # ordinal variables
    ordinal_df = df.select_dtypes(include = ['int64', 'Int64'])

    # select col modes
    modes = ordinal_df.mode() # mode of each col
    modes = modes.iloc[0] # select modes only

    # calc quasi percentage
    quasi_percentages = (ordinal_df == modes).sum() / len(ordinal_df) # how quasi a col is

    return quasi_percentages

# drop quasi features
def drop_quasi_features(df, threshold):

    # calculate quasi percentage
    quasi_percentages = quasi_percentage(df)

    # 0.995 is about 3 observations
    exceeding_threshold_columns = quasi_percentages[quasi_percentages > threshold].index
    df = df.drop(columns = exceeding_threshold_columns)
    return df

######################################################################################################################## correlation functions

def select_future_features(df):
    EndDesc_cols = [col for col in df.columns if 'EndDesc' in col]
    #EndDescShort_cols = [col for col in df.columns if 'EndDescShort' in col] # none
    future_vars = df.drop(columns = ['ReliableChangeDesc', 'Recovery', 'ReliableRecovery'] + EndDesc_cols)
    return future_vars

def generate_corr_matricies(df, load_matrices):

    file_path = '/content/drive/MyDrive/Data/Dissertation_Data/'

    if not load_matrices:
        # select explanatory variables, removing info on future
        future_df = select_future_features(df)

        # create correlation matrices
        kendall_corr = future_df.corr(method='kendall').abs()
        spearman_corr = future_df.corr(method='spearman').abs()
        kendall_corr.to_csv(file_path + 'kendall_corr.csv') # save
        spearman_corr.to_csv(file_path + 'spearman_corr.csv') # save

    else:
        kendall_corr = pd.read_csv(file_path + 'kendall_corr.csv', index_col=0)
        spearman_corr = pd.read_csv(file_path + 'spearman_corr.csv', index_col=0)

    return kendall_corr, spearman_corr

def plot_correlation_matrices(corr_matricies):

    plt.figure(figsize=(8, 3))

    plt.subplot(121)
    sns.heatmap(corr_matricies[0], annot = False, cmap = 'flare', fmt = ".2f", xticklabels = False, yticklabels = False)
    plt.title('Kendall Correlation Matrix')

    plt.subplot(122)
    sns.heatmap(corr_matricies[1], annot = False, cmap = 'flare', fmt = ".2f", xticklabels = False, yticklabels = False)
    plt.title('Spearman Correlation Matrix')

    plt.tight_layout()
    plt.show()

# count correlated variable pairs above each threshold
def count_correlated_pairs(correlation_matrix, thresholds):

    counts = {}
    for threshold in thresholds:
        correlated_vars = (correlation_matrix.abs() > threshold)
        # get upper triangle
        np.fill_diagonal(correlated_vars.values, False)
        upper_triangle = correlated_vars.values[np.triu_indices_from(correlated_vars, k=1)]
        count = np.sum(upper_triangle)
        counts[threshold] = count
    return counts

def correlated_features_above_treshold(matricies, thresholds):

    print('Kendall Correlation:')
    kendall_counts = count_correlated_pairs(matricies[0], thresholds)
    for threshold, count in kendall_counts.items():
        print(f'{threshold * 100}%: {count}')

    print('Spearman Correlation:')
    spearman_counts = count_correlated_pairs(matricies[1], thresholds)
    for threshold, count in spearman_counts.items():
        print(f'{threshold * 100}%: {count}')

def drop_corr_cols(df, corr_matrix, threshold, ignore=True):

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]

    try:
        df = df.drop(corr_cols, axis=1)
    except KeyError as e:
        if ignore:
            #print(f"Ignoring KeyError: {e}")
            pass
        else:
            raise

    return df

def remove_corr_features(df, load_matrices, threshold):

    kendall_corr, spearman_corr = generate_corr_matricies(df, load_matrices)

    # remove highly correlated variables
    df = drop_corr_cols(df, kendall_corr, threshold)
    df = drop_corr_cols(df, spearman_corr, threshold)

    return df

######################################################################################################################## missing value functions

def drop_missing_values_col(df, threshold):

    missing_value_percentage = df.isna().mean(axis=0) # cols

    drop_cols = missing_value_percentage[missing_value_percentage > threshold].index
    df = df.drop(drop_cols, axis=1)

    return df

def drop_missing_values_row(df, threshold):

    missing_value_percentage = df.isna().mean(axis=1) # rows

    drop_cols = missing_value_percentage[missing_value_percentage > threshold].index
    df = df.drop(drop_cols, axis=0)

    return df

######################################################################################################################## scaling functions

def standardise(df):
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df

######################################################################################################################## imputation functions

def impute_data(df, method, n_neighbours=None, max_iter=None):

    original_dtypes = df.dtypes.to_dict()

    #EndDescShort_cols = [col for col in df.columns if 'EndDescShort' in col] # none
    EndDesc_cols = [col for col in df.columns if 'EndDesc' in col]
    nontarget_df = df.drop(columns = ['ReliableChangeDesc', 'Recovery', 'ReliableRecovery'] + EndDesc_cols)

    datetime_cols = nontarget_df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_cols:
        nontarget_df[col] = pd.to_numeric(nontarget_df[col])

    if method == 'none':
        nontarget_mat = nontarget_df.values

    elif method == 'mean':
        imputer = SimpleImputer(strategy='mean')
        nontarget_mat = imputer.fit_transform(nontarget_df) # *numpy array*

    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
        nontarget_mat = imputer.fit_transform(nontarget_df)

    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbours)
        nontarget_mat = imputer.fit_transform(nontarget_df)

    elif method == 'iterative':
        imputer = IterativeImputer(max_iter=max_iter, random_state=2001)
        nontarget_mat = imputer.fit_transform(nontarget_df)

    nontarget_df.iloc[:, :] = nontarget_mat.tolist()

    # add imputed data to df
    for col in nontarget_df.columns:
        df[col] = nontarget_df[col]

    if method == 'none':
        pass

    else:
        for col, dtype in original_dtypes.items():
            if dtype == 'datetime64[ns]':
                pass # keep as float
            else:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    # *convert to original dtype*
                    df[col] = df[col].round().astype(int)
                    df[col] = df[col].astype(dtype)

    return df

######################################################################################################################## feature engineering functions

def create_date_features(df):

    # df['ReferralYear'] = df['ReferralDate'].dt.year
    # df['ReferralMonth'] = df['ReferralDate'].dt.month
    # df['ReferralWeek'] = df['ReferralDate'].dt.isocalendar().week
    # df['ReferralDay'] = df['ReferralDate'].dt.day
    # df['ReferralHour'] = df['ReferralDate'].dt.hour
    # df['ReferralWeekDay'] = df['ReferralDate'].dt.dayofweek
    # df['ReferralYearDay'] = df['ReferralDate'].dt.dayofyear

    df['YearofQuestionnaire'] = df['DateOfQuestionnaire'].dt.year
    df['MonthofQuestionnaire'] = df['DateOfQuestionnaire'].dt.month
    df['WeekofQuestionnaire'] = df['DateOfQuestionnaire'].dt.isocalendar().week
    df['DayofQuestionnaire'] = df['DateOfQuestionnaire'].dt.day
    df['HourofQuestionnaire'] = df['DateOfQuestionnaire'].dt.hour
    df['WeekDayofQuestionnaire'] = df['DateOfQuestionnaire'].dt.dayofweek
    df['YearDayofQuestionnaire'] = df['DateOfQuestionnaire'].dt.dayofyear

    return df

def feature_engineering(df):

    df = create_date_features(df)

    return df

######################################################################################################################## feature imporatance functions

def find_important_features(df, k_features, target='Recovery'):

    """
    k is the number of features each feature selection method should select.
    NOT the number of features returned

    """

    if k_features == None:
        important_features = df.columns

    else:
        features = df.dropna()
        EndDesc_cols = [col for col in df.columns if 'EndDesc' in col]
        explanatory = features.drop(['Recovery', 'ReliableRecovery', 'ReliableChangeDesc'] + EndDesc_cols, axis = 1)
        target = features[target]

        # fishers score
        kbest_fisher = SelectKBest(score_func=f_classif, k=k_features)
        selected_fisher = kbest_fisher.fit_transform(explanatory, target)
        index_fisher = kbest_fisher.get_support(indices=True)
        names_fisher = explanatory.columns[index_fisher]

        # mutual information gain
        def mutual_info_classif_wseed(X, y):
            return mutual_info_classif(X, y, random_state=11)
        kbest_gain = SelectKBest(score_func=mutual_info_classif_wseed, k=k_features)
        selected_gain = kbest_gain.fit_transform(explanatory, target)
        index_gain = kbest_gain.get_support(indices=True)
        names_gain = explanatory.columns[index_gain]

        important_combined = set(names_fisher).union(set(names_gain))
        important_features = list(important_combined)

    return important_features

def select_important_features(df, k_features, target='Recovery'):

    important_features = find_important_features(df, k_features, target='Recovery')

    EndDesc_cols = [col for col in df.columns if 'EndDesc' in col]
    df = df[['Recovery', 'ReliableRecovery', 'ReliableChangeDesc'] + EndDesc_cols + important_features]

    return df

######################################################################################################################## Preparation Function (final)

def Prepare_Data(df,
                 quasi_thresh=1.0,
                 corr_thresh=1.0,
                 load_matrices=True,
                 col_thresh=1.0,
                 row_thresh=1.0,
                 imputation_method='knn',
                 n_neighbours=10,
                 max_iter=10,
                 k_features=200):

    df = drop_duplicate_features(df)
    df = replace_outliers(df)
    df = drop_const_features(df)
    df = drop_quasi_features(df, quasi_thresh)
    df = remove_corr_features(df, corr_thresh, load_matrices)
    df = drop_missing_values_col(df, col_thresh)
    df = drop_missing_values_col(df, row_thresh)
    #df = standardise(df)
    df = impute_data(df, imputation_method, n_neighbours, max_iter)
    #df = feature_engineering(df)
    df = select_important_features(df, k_features)

    return df
