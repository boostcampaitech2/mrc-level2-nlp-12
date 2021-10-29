from datasets import load_from_disk
 

def preprocess(df) -> list:
    '''
    answer를 기준으로 좌/우로 context를 나누고 전처리를 진행
    answer에 어떠한 영향도 주지 않기 위함.

    df: Pandas Dataframe

    [return] : preprocessed context list
    '''
    context_list = []
    for i in range(len(df)):
        start_idx = df.iloc[i]['answers']['answer_start'][0]
        len_answer = len(df.iloc[i]['answers']['text'][0])
        answer = df.iloc[i]['context'][start_idx:start_idx+len_answer]
        context1 = df.iloc[i]['context'][:start_idx]
        context2 = df.iloc[i]['context'][start_idx+len_answer:]

        context1 = replace_chars(context1)
        context2 = replace_chars(context2)

        context_list.append(context1 + answer + context2)
            
    return context_list

def replace_chars(context):
    cotext = context.replace('\n', ' ')
    context = context.replace('\\n', ' ')
    context = context.replace('  ', ' ')
    return context


datasets = load_from_disk('../data/train_dataset')
train_df = datasets['train'].to_pandas()
val_df = datasets['validation'].to_pandas()  

train_context_list = preprocess(train_df)
val_context_list = preprocess(val_df)

train_df.drop('context', axis=1, inplace=True)
val_df.drop('context', axis=1, inplace=True)

train_df = train_df[['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']]
val_df = val_df[['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__']]
