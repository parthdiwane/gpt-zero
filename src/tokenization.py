import regex as re

class tokenization:
    def __init__(self, corpus):
        self.corpus = corpus

    def pre_tokenization(self, text):
        # contractions, words, numbers, non-english words and numebrs, sucessvive whitespaces
        return re.findall(r'\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+', text)

    def intialize(self, pre_tokenized_text):
        initialization = []
        for word in pre_tokenized_text:
            addition = []
            for char in word:
                if char == " ":
                    addition.append("</w>")
                else:
                    addition.append(char)
            initialization.append(tuple(addition))
        return initialization
    
    def find_most_freq(self, text):
        freq_map = {}
        for word in text:
            for i in range(len(word) - 1):
                subset = word[i : i + 2]
                if subset in freq_map:
                    freq_map[subset] += 1
                else:
                    freq_map[subset] = 1
        most_freq = max(freq_map, keys=freq_map.get)
        return most_freq

    def replace(self, text, subset):
        for j in range(len(text)):
            word = text[j]
            for i in range(len(word) - 1):
                if word[i : i + 2] == subset:
                    text[j] = word[0:i] + subset + word[i + 2:]
        return text
        
    def bpe(self, text, k):
        pre_token = self.pre_tokenization(text=text)
        initialization = self.intialize(text)
    
        vocab = set()

        # add stuff to vocab
        for i in range(k):
            curr_most_freq = self.find_most_freq(initialization)
            vocab.add(curr_most_freq)
            initialization = self.replace(initialization, curr_most_freq)

        
             
                 
        
if __name__ == "__main__":
    test = tokenization("hi")
    print(test.pre_tokenization("hello world"))
    print(test.bpe("hello world", 10))