import urllib.request
import re
import tiktoken

# Our goal is to tokenize this 20479 character into individial workds
# and special characters that can be turned into embeddings for LLM training


# Download the text file from the provided URL and save it locally
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
    "ch02/01_main-chapter-code/the-verdict.txt"
)
file_path = "01_fundamentals/tokenizing_text/the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# Load the  the-verdict.txt file
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
print("Total number of character:", len(raw_text))

result = re.split(r'(\s|[,.:;?_!"()\']|--)', raw_text)
# Remove empty strings and whitespace-only strings
preprocessed = [item for item in result if item.strip()]
print(
    "After removing empty strings and whitespace-only strings:",
    preprocessed[0:100],
)

# Converting token into token_IDs
# Create a vocabulary of unique tokens
soted_text = sorted(set(preprocessed))
vocal_size = len(soted_text)
print("Vocabulary size:", vocal_size)
vocab = {token: id for id, token in enumerate(soted_text)}
# When converting LLM outputs to text, we need bidirectional mapping:
# token_ID to token and token to token_ID


# Implement a simple text tokenizer
# 1 Stores the vocabulary as a class attribute for encode/decode
# 2 Creates inverse vocabulary (token IDs to text tokens)
# 3 Processes input text into token IDs
# 4 Converts token IDs back into text
# 5 Removes spaces before specified punctuation


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        #  vocab is something that looks like this
        # ('!', 0)
        # ('"', 1)
        # ("'", 2)
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        tokens = re.split(r'(\s|[,.:;?_!"()\']|--)', text)
        preprocessed = [item for item in tokens if item.strip()]
        token_ids = [self.str_to_int[t] for t in preprocessed]
        return token_ids

    def decode(self, token_ids):

        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = re.sub(
            r'\s+([,.?!"()\'])', r"\1", text
        )  # ' can not be recognized, use \'
        return text


# Calling tokenizer.decode on token IDs
tokenizer = SimpleTokenizerV1(vocab)
ids = tokenizer.encode(raw_text)
# print(ids)
# print(tokenizer.decode(ids))

# Chapter 2.4: Adding Special context tokens
# Modify vocab and tokenizer to handle unknown words and document
# boundaries: <|unk|>, <|endoftext|>
# We can add token between unrelated texts
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])
vocab = {token: id for id, token in enumerate(all_tokens)}
# print(len(vocab.items()))


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab  # key: int
        self.int_to_str = {i: k for k, i in vocab.items()}

    def encode(self, text):
        tokens = re.split(r'(\s|--|[;:,.?!_"()\'])', text)
        preprocessed = [item.strip() for item in tokens if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        token_ids = [self.str_to_int[t] for t in preprocessed]
        return token_ids

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))


# Additional special tokens used in some LLMs:
#
# [BOS] (beginning of sequence) - Marks the start of a text.
#       Signifies to the LLM where content begins.
#
# [EOS] (end of sequence) - Positioned at the end of a text.
#       Useful when concatenating multiple unrelated texts,
#       similar to <|endoftext|>. Indicates where one text ends
#       and the next begins.
#
# [PAD] (padding) - When training LLMs with batch sizes > 1,
#       texts may have varying lengths. Shorter texts are extended
#       or "padded" using the [PAD] token to match the longest text.
#
# Excerpt From: Build a Large Language Model (From Scratch)
# by Sebastian Raschka


# In GPT models, we dont use '<|unk|>',
# instead we use a concept called byte pair encoding
# Chapter 2.5 Byte Pair Encoding (BPE)
# The algorithm underlying BPE breaks down words that aren't in its
# predefined vocabulary into smaller subword units or even individual
# characters, enabling it to handle out-of-vocabulary words.
# Thanks to BPE, if the tokenizer encounters an unfamiliar word during
# tokenization, it can represent it as a sequence of subword tokens
# or characters.

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode(integers))
