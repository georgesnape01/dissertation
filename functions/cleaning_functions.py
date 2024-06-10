def load_data():

    drive.mount("/content/drive") # access google drive folder

    file_path = '/content/drive/MyDrive/Data/Dissertation_Data/'
    df = pd.read_csv(file_path + 'mental_health.csv', delimiter = ',') # csv
    raw_df = df.copy()

    return df, raw_df

def manage_df(df, name, option):

    diss_path = '/content/drive/MyDrive/Data/Dissertation_Data/'

    if option == 'save':
        df.to_csv(file_path + name + '.csv')
        print('Saved!')

    elif option == 'load':
        df = pd.read_csv(file_path + name + '.csv', index_col=0)
        return df
    
def rename_features(df): # non-questionnaire features

    df.rename(columns = {
        'Unnamed: 0': 'CaseID',
        'IAPTus_Num': 'ClientID',
        'Referral Date': 'ReferralDate',
        'Age_ReferralRequest_ReceivedDate': 'AgeAtReferralRequest',
        'EthnicDescGroupCode': 'EthnicCode',
        'EthnicCategoryGroupShortCode': 'EthnicCodeShort',
        'GenderIdentity': 'Gender',
        'SexualOrientationDesc': 'SexualOrientation',
        'EndDescGroupShort': 'Treated',
        'AllocatedCareProfNameCode': 'TherapistID',
        'JobTitleCode': 'ExperienceLevel',
        'Days to first assessment': 'DaystoAssessment',
        'Days to first treatment': 'DaystoTreatment',
        'CountOfAttendedCareContacts': 'CareContacts',
        'RecoveryDesc': 'Recovery',
        'ReliableRecoveryDesc': 'ReliableRecovery',
        'Date': 'DateOfQuestionnaire'},
        inplace = True)

    return df

def create_referral_count(df):
    def count_referrals(col):
        if '_1' in col:
            return 1
        elif '_2' in col:
            return 2
        elif '_3' in col:
            return 3
        elif '_4' in col:
            return 4
        elif '_5' in col:
            return 5
        else:
            return 1
    df.insert(2, "ReferralCount", df['ClientID'].apply(count_referrals)) # introduce next to ClientID
    return df

def clean_client_id(df):
    for text in ['_1', '_2', '_3', '_4']:
        df['ClientID'] = df['ClientID'].str.replace(text, '') # remove ending
    df['ClientID'] = pd.to_numeric(df['ClientID'])
    return df

def convert_features_to_datetime(df):
    df['ReferralDate'] = pd.to_datetime(df['ReferralDate'], format = '%d/%m/%Y')
    df['DateOfQuestionnaire'] = pd.to_datetime(df['DateOfQuestionnaire'], format = '%d/%m/%Y')
    return df

def convert_float_features_to_int(df):
    df['EthnicCode'] = df['EthnicCode'].astype('Int64') # Int deals with NaNs, int does not
    df['EthnicCodeShort'] = df['EthnicCodeShort'].astype('Int64')
    df['TherapistID'] = df['TherapistID'].astype('Int64')
    df['ExperienceLevel'] = df['ExperienceLevel'].astype('Int64')
    return df

def map_features(df):

    Gender_map = {
        'CHANGE ME': np.nan,
        'X': np.nan}
    df['Gender'] = df['Gender'].replace(Gender_map).astype('Int64')

    Treated_map = {
        'Seen and treated': 1,
        'Seen but not treated': 0}
    df['Treated'] = df['Treated'].replace(Treated_map).astype('Int64')

    ReliableChangeDesc_map = {
        'Reliable improvement': 2, # what about(-1, 0, 1)?
        'No reliable change': 1,
        'Reliable deterioration': 0,
        'Not applicable': np.nan}
    df['ReliableChangeDesc'] = df['ReliableChangeDesc'].replace(ReliableChangeDesc_map).astype('Int64')

    Recovery_map = {
        'At recovery': 1,
        'Not at recovery': 0,
        'Not applicable': np.nan}
    df['Recovery'] = df['Recovery'].replace(Recovery_map).astype('Int64')

    ReliableRecovery_map = {
        'Reliable recovery': 1,
        'No reliable recovery': 0,
        'Not applicable': np.nan}
    df['ReliableRecovery'] = df['ReliableRecovery'].replace(ReliableRecovery_map).astype('Int64')

    return df

def one_hot_encode_features(df):

    EndDesc_cols = pd.get_dummies(df['EndDesc'], prefix = 'EndDesc')
    EndDesc_index = df.columns.get_loc('EndDesc')
    df = pd.concat([df.iloc[:, :EndDesc_index + 1], EndDesc_cols, df.iloc[:, EndDesc_index + 1:]], axis = 1)
    df = df.drop(columns = ['EndDesc'])

    EndDescShort_cols = pd.get_dummies(df['EndDescShort'], prefix = 'EndDescShort')
    EndDescShort_index = df.columns.get_loc('EndDescShort')
    df = pd.concat([df.iloc[:, :EndDescShort_index + 1], EndDescShort_cols, df.iloc[:, EndDescShort_index + 1:]], axis = 1)
    df = df.drop(columns = ['EndDescShort'])

    return df

def convert_to_int_features(df):
    int_cols = ['SexualOrientation', 'DaystoAssessment', 'DaystoTreatment', 'CareContacts']
    for col in int_cols:
        df[col] = df[col].astype('Int64')
    return df

def plot_features(df):

    plot_cols = 4
    plot_rows = len(df.columns)//4 + 1

    plt.figure(figsize = (plot_cols*3, plot_rows*3))

    for i, col in enumerate(df.columns, start = 1):
        if df[col].dtype in ['int64', 'float64', 'Int64', 'datetime64[ns]', 'bool']:
            data = df[col].dropna()
            if df[col].dtype == 'bool':  # convert boolean to integers
                data = data.astype(int)
            plt.subplot(plot_rows, 4, i)
            plt.hist(data, bins = 20, color='darkgrey', edgecolor='white')
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel('Frequency')

    plt.suptitle('Data Plots', y = 1, fontsize = 24)
    plt.tight_layout()
    plt.show()

def preprocess_nonques_features(df):

    df = rename_features(df)
    df = create_referral_count(df)
    df = clean_client_id(df)
    df = convert_features_to_datetime(df)
    df = convert_float_features_to_int(df)
    df = map_features(df)
    df = one_hot_encode_features(df)
    df = convert_to_int_features(df)

    return df

def preprocess_ques_features(df):

    # convert all variables to float
    for col in df.columns[27:]:
        df[col] = pd.to_numeric(df[col], errors = 'coerce') # (I think this also removes '.')

    item_cols = df.columns.str.contains('Item').tolist()
    item_cols = df.columns[item_cols]
    for col in item_cols:
        df[col] = df[col].apply(lambda x: x if pd.isna(x) or x.is_integer() else np.nan) # NaN non-integer values
        df[col] = df[col].astype('Int64') # convert to int

    thresh_cols = df.columns.str.contains('Threshold').tolist()
    thresh_cols = df.columns[thresh_cols]
    for col in thresh_cols:
        df[col] = df[col].apply(lambda x: x if pd.isna(x) or x.is_integer() else np.nan)
        df[col] = df[col].astype('Int64')

    total_cols = df.columns.str.contains('Total').tolist()
    total_cols = df.columns[total_cols]
    for col in total_cols:
        df[col] = df[col].apply(lambda x: x if pd.isna(x) or x.is_integer() else np.nan)
        df[col] = df[col].astype('Int64')

    # bool cols (all)
    bool_cols = df.select_dtypes(include = bool)
    for col in bool_cols:
        df[col] = df[col].astype('Int64')

    return df

def Clean_Data(df):

    # clean non-questionnaire data
    df = preprocess_nonques_features(df)

    # clean questionnaire data
    df = preprocess_ques_features(df)

    return df
