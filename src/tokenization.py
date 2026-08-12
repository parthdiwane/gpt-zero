import regex as re
import numpy as np

class tokenization:
    def __init__(self, corpus):
        self.corpus = corpus

    def from_npz(cls, path, keys=None, join=" "):
        return cls(cls.load_npz(path, keys=keys, join=join))

    def decode_entry(entry):
        # npz round-trips words as np.str_, bytes, or object depending on how it was saved
        if isinstance(entry, bytes):
            return entry.decode("utf-8")
        return str(entry)

    def load_npz(path, keys=None, join=" "):
        words = []
        with np.load(path, allow_pickle=True) as archive:
            selected = archive.files if keys is None else keys
            if isinstance(selected, str):
                selected = [selected]
            for key in selected:
                array = archive[key]
                if array.dtype.kind not in ("U", "S", "O"):
                    raise ValueError(f"'{key}' in {path} holds {array.dtype}, not text")
                if array.ndim == 0:
                    words.append(tokenization.decode_entry(array.item()))
                else:
                    for entry in array.ravel():
                        words.append(tokenization.decode_entry(entry))
        return join.join(words)


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
        self.vocab_id = {}
        current_id = 1
        for i in range(k):
            most_freq_chars = self.find_most_freq(pre_tokenized_text=pretokenized_sentence) # finding the most frequent characters

            self.vocab.append(most_freq_chars) # add tings to vocab
            self.vocab_id[most_freq_chars] = current_id
            current_id += 1

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
        tokenized_text_id = []
        for token in tokenized_text:
            tokenized_text.append(self.token_id[token])
    
        return tokenized_text, tokenized_text_id

    def token_id(self):
        return self.token_id

    