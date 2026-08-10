import regex as re

class tokenization:
    def __init__(self, corpus):
        self.corpus = corpus


    def pre_tokenize(self, text):
        pre_tokenized_text = re.findall(r'\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+', text) # regex for gpt2 i think
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
    
    def find_most_freq(self, pre_tokenized_text): # pre-token is a list of tuples
        freq_map = {}
        for word in pre_tokenized_text:
            for i in range(len(word) - 1):
                subset = word[i : i + 2]
                current_pair = ""
                for char in subset:
                    current_pair += char
                if current_pair in freq_map:
                    freq_map[current_pair] += 1
                else:
                    freq_map[current_pair] = 1
        most_freq = max(freq_map, key=freq_map.get)
        return most_freq

    def replace(self, text, subset):
        for words in range(len(text)):

            current_word = list(text[words])
            n = len(current_word)
            i = 0

            replaced_word = []

            while i < n:
                if i < n - 1 and "".join(current_word[i : i + 2]) == subset:
                    replaced_word.append("".join(current_word[i : i + 2]))
                    i += 2
                else:
                    replaced_word.append(current_word[i])
                    i += 1
            text[words] = tuple(replaced_word)
        return text


    def bpe_learner(self, text, k):
        pretokenized_sentence = self.pre_tokenize(text)

        self.vocab = []

        for i in range(k):
            most_freq_chars = self.find_most_freq(pre_tokenized_text=pretokenized_sentence)
            self.vocab.append(most_freq_chars)
            pretokenized_sentence = self.replace(pretokenized_sentence, most_freq_chars)


    def bpe_segmenter(self, text):
        pretoken = self.pre_tokenize(text=text)
        for v in self.vocab:
            for word_index in range(len(pretoken)):
                word = pretoken[word_index]
                new_word = ""
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and v == word[i : i + 2]:
                        new_word += v
                    else:
                        new_word += word[i]
                    i += 1
                pretoken[word_index] = new_word
        tokenized_text = pretoken
        return tokenized_text



            

    

        
if __name__ == "__main__":
    test = tokenization("hi")
    test.bpe_learner("hello world", 5)
    print(test.bpe_segmenter("hello world"))

    