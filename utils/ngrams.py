def remove_exact_substring_matches(dictionary):
            keys_to_remove = []
            keys = list(dictionary.keys())
            for i in range(len(keys)):
                current_key = keys[i]
                for j in range(len(keys)):
                    if i != j:  
                        other_key = keys[j]
                        if current_key in other_key:
                            keys_to_remove.append(current_key)
                            break 
            for key in keys_to_remove:
                del dictionary[key]
            return dictionary
        

def remove_exact_unwanted_Keys(data):
    to_delete = []

    for key in data:
        words = key.split()
        lengths = [len(word) for word in words]
        
        if all(length <= 3 for length in lengths):
            to_delete.append(key)
        

    for key in to_delete:
        del data[key]
    return data

def frequent_ngrams(words, n):
    try:
        
        
        # print(words)
    
        ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


        ngram_counts = {}
        for gram in ngrams:
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1

    
        frequent_ngrams = {
            gram: count
            for gram, count in ngram_counts.items()
            if count >= 2
        }

        return frequent_ngrams
    except Exception as e:
        print(f"frequent_{n}grams", str(e))
        return {}    
    

def frequent_dynamic_ngrams(text, combined_data):
    try:
        words = text.split()
        words = [word for word in words if word not in combined_data]
        all_frequent_ngrams = {}
        n = 4  

        while True:
            ngram_freq = frequent_ngrams(words, n)
            # print(ngram_freq)
            if not ngram_freq:
                break
            all_frequent_ngrams[f"{n}-gram"] = ngram_freq
            n += 1

        merged_dict = {}
        for ngram_dict in all_frequent_ngrams.values():
            merged_dict.update(ngram_dict)

        merged_dict1 = remove_exact_substring_matches(merged_dict)
        remove_2 = remove_exact_unwanted_Keys(merged_dict1)
        return remove_2
    
    except Exception as e:
        print("frequent_dynamic_ngrams", str(e))
        return {}