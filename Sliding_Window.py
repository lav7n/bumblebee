
#from Preprocessing import training_notes,training_duration

def sliding_window(training_notes,training_duration):
    train = []
    label = []
    window_size = 10
    for x,y in zip(training_notes,training_duration):
        
        if len(x)>window_size:
                for index in range(len(x)-1-window_size):
                    in_1 = x[index:index+window_size] + ['NA'] + y[index:index+window_size]
                    out_1 = x[index+1:index+1+window_size] + ['NA'] + y[index+1:index+1+window_size]
                    train.append(in_1)
                    label.append(out_1)
        else:
            pass
    return train,label