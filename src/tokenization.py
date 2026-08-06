import regex as re

class tokenization:
    def __init__(self, corpus):
        self.corpus = corpus

    def pre_tokenization(self, text):
        # contractions, words, numbers, non-english words and numebrs, sucessvive whitespaces
        return re.findall(r'\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+', text)

    def bpe(self, text):
        pre_token = self.pre_tokenization(text=text)
        


if __name__ == "__main__":
    test = tokenization("hi")
    print(test.pre_tokenization("hello world"))