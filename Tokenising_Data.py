import pandas as pd
from sklearn.model_selection import train_test_split as tts
#from Sliding_Window import train,label

def tokenising_data(train,label):
    train_df = pd.DataFrame({'input_text':[x for x in train],'target_text':[x for x in label]})
    print(train_df['input_text'].iloc[:10])
    map_dict = {}
    unique_values = sorted(train_df['input_text'].explode().unique())
    for index, value in enumerate(unique_values):
        index = index%26
        map_dict[value] = chr(index + 65)
        print(index)

    #map_dict = {y:string.ascii_uppercase[x] for x,y in enumerate(sorted(train_df['input_text'].explode().unique()))}

    train_df = train_df.applymap(lambda x: ' '.join([map_dict[y] for y in x]))

    training, validation = tts(train_df)
    training = training.sample(len(training))
    return training,validation