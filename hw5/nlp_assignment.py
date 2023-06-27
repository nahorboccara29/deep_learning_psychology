import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Question 1

class RNNLayer(nn.Module):
    def __init__(self, n):
        super(RNNLayer, self).__init__()
        self.n = n
        self.W = nn.Parameter(torch.randn(2*n, n))
    
    def forward(self, x, h):
        # Concatenate input and history vectors
        x_h = torch.cat([x, h], dim=-1)  # dim=-1 indicates the last dimension, this assumes the input is (batch_size, n)
        
        # Linear transformation with weight matrix W
        output = torch.matmul(x_h, self.W)
        
        return output
    
# Question 2

'''
In the context of recurrent neural networks (RNNs), h_1 and h_t would represent the hidden state of the network after processing the first word and the t-th word of a
sequence, respectively.

h_1: After processing the first word of a sequence, the hidden state h_1 represents the network's "memory" of that first word. It is calculated based on the initial
hidden state (usually initialized to zeros) and the first word in the sequence. This hidden state captures the information in the first word and is used as part of
the input when processing the second word.

h_t: After processing t words of a sequence, the hidden state h_t represents the "memory" of the network after seeing the first t words. This state is a function of
all previous hidden states and the corresponding input words. This means that h_t captures some information about all the previous words in the sequence, with newer
words typically having more influence than older ones. However, basic RNNs can struggle to maintain information from older inputs due to the "vanishing gradients"
problem, which makes them less effective at capturing long-term dependencies.

In a practical application like language modeling or sentiment analysis, h_t could be used at each step to predict the next word in the sequence, or the final h_t
could be used to make a prediction about the entire sequence (such as its overall sentiment).

Keep in mind that while this gives you a high-level view of what h_1 and h_t represent in an RNN, the actual behavior and effectiveness of the network can depend on
many factors, including how the network is trained and the specific type of RNN used (LSTM, GRU, etc.).
'''

# Question 3

class SelfAttention(nn.Module):
    def __init__(self, n):
        super(SelfAttention, self).__init__()
        self.n = n
        self.W_Q = nn.Parameter(torch.randn(n, n))
        self.W_K = nn.Parameter(torch.randn(n, n))
        self.W_V = nn.Parameter(torch.randn(n, n))

    def forward(self, x):
        # Step 1: Calculate Q, K, V
        Q = torch.matmul(x, self.W_Q)
        K = torch.matmul(x, self.W_K)
        V = torch.matmul(x, self.W_V)

        # Step 2: Calculate attention map
        K_T = torch.transpose(K, -2, -1)
        QK_T = torch.matmul(Q, K_T)

        # Attention map after applying softmax row-wise
        attn_map = F.softmax(QK_T, dim=-1)

        # Step 3: Calculate output representations
        output = torch.matmul(attn_map, V)

        return output
    
# Question 4

class SelfAttention(nn.Module):
    def __init__(self, n):
        super(SelfAttention, self).__init__()
        self.n = n
        self.W_Q = nn.Parameter(torch.randn(n, n))
        self.W_K = nn.Parameter(torch.randn(n, n))
        self.W_V = nn.Parameter(torch.randn(n, n))

    def forward(self, x):
        # Step 1: Calculate Q, K, V
        Q = torch.matmul(x, self.W_Q)
        K = torch.matmul(x, self.W_K)
        V = torch.matmul(x, self.W_V)

        # Step 2: Calculate attention map
        K_T = torch.transpose(K, -2, -1)
        QK_T = torch.matmul(Q, K_T)

        # Attention map after applying softmax row-wise
        attn_map = F.softmax(QK_T, dim=-1)

        # Step 3: Calculate output representations
        output = torch.matmul(attn_map, V)

        return output, attn_map  # Return both output and attention map
    
# Usage example:
n = 100  # size of input vector
self_attn = SelfAttention(n)

# Dummy input vector for demonstration purposes
x = torch.randn(1, n)  # input vector (assuming batch size is 1)

output, attn_map = self_attn(x)

# Plot the attention map as a heatmap
attn_map_np = attn_map.detach().numpy()  # Convert tensor to numpy for plotting
plt.figure(figsize=(10, 10))
sns.heatmap(attn_map_np, cmap='viridis')
plt.show()

# Question 5

'''
a. Each pixel in the heatmap represents the attention weight for a pair of elements in the input sequence. In the context of self-attention, it indicates how much an
element should "attend to" another element when producing the output.

b. Each row in the attention map corresponds to an element in the input sequence, and shows the attention weights for that element with respect to all other elements.
In other words, it represents how much this specific element should consider other elements when updating its state.

c. When you sum over each row, you get 1. This is because the attention weights for each input are calculated using a softmax function, which normalizes the weights
so that they sum to 1. This means that each input's output representation is a weighted sum of the input representations, with weights that sum to 1.

d. The attention map is not necessarily symmetrical. The attention weight that a given input element i gives to input element j doesn't necessarily equal the weight
that j gives to i. This is because the weights depend on the specific interaction of the query and key for each pair, which changes depending on the order of the pair.
This means that a row does not necessarily have the same meaning as the corresponding column.

e. When you multiply each row from the attention map with V, you're computing a weighted sum of the value vectors. Each output element is thus a weighted combination
of all input elements, where the weights represent the degree of "attention" that element should pay to each other element. This allows the network to focus on the most relevant parts of the input for each output element, increasing its ability to model complex patterns and relationships.

f. As for the neuroscience perspective, the concepts of Q (query), K (key), and V (value) don't have direct equivalents in terms of brain function. However, the
general principle of attention—that is, focusing more processing resources on certain inputs while ignoring others—is a fundamental aspect of how brains manage
information. In this sense, self-attention mechanisms in neural networks are a highly simplified and abstracted version of the attention processes that occur in the
brain. Comparisons between neural network attention and human attention can lead to interesting discussions and research, but should be made with the understanding
that the two are different in many fundamental ways.
'''

# Question 6

# a
word_vectors = {
    'good': torch.randn(1, 100),
    'not': torch.randn(1, 100),
    'too': torch.randn(1, 100),
    'bad': torch.randn(1, 100),
}

# b
phrase1 = ['good', 'not', 'too', 'bad']
phrase1_vectors = torch.cat([word_vectors[word] for word in phrase1])

self_attn = SelfAttention(100)
embeddings1, _ = self_attn(phrase1_vectors)

# c
phrase2 = ['bad', 'not', 'too', 'good']
phrase2_vectors = torch.cat([word_vectors[word] for word in phrase2])

embeddings2, _ = self_attn(phrase2_vectors)

# d
'''
The embeddings of the words in the phrases "good, not too bad" and "bad, not too good" are different even though they contain the same words. This is because the
self-attention layer pays attention to the order of the words. It assigns different weights to each word based on its position in the sequence and its relationship
with other words in the sequence.
'''

# e
word_vectors_encoded = {
    word: vector + math.sin(i) for i, (word, vector) in enumerate(word_vectors.items())
}

# f
phrase1_vectors_encoded = torch.cat([word_vectors_encoded[word] for word in phrase1])
phrase2_vectors_encoded = torch.cat([word_vectors_encoded[word] for word in phrase2])

embeddings1_encoded, _ = self_attn(phrase1_vectors_encoded)
embeddings2_encoded, _ = self_attn(phrase2_vectors_encoded)

# g
'''
Yes, the representations will be different now due to the added positional encoding. Temporal encoding is crucial in NLP tasks because the meaning of words often
depends on their order in a sentence. Without some form of temporal encoding, models like the self-attention layer would treat sentences as a bag of words, where the
order of words doesn't matter. This would prevent the model from capturing important syntactic and semantic relationships between words. By adding temporal encoding,
we allow the model to take into account the order of words and capture these important relationships.

Please note that these are basic examples. In a real-world scenario, the word vectors would be learned from data (not randomly initialized), and more complex methods
of temporal encoding would be used. For instance, in the Transformer model, a more complex form of positional encoding is used, which uses both sine and cosine
functions to create a dense encoding of positions.
'''

# Question 7

class EncoderDecoderAttention(nn.Module):
    def __init__(self, n):
        super(EncoderDecoderAttention, self).__init__()
        self.n = n
        self.W_Q = nn.Parameter(torch.randn(n, n))
        self.W_K = nn.Parameter(torch.randn(n, n))
        self.W_V = nn.Parameter(torch.randn(n, n))

    def forward(self, decoder_input, encoder_output):
        # Step 1: Calculate Q, K, V
        Q = torch.matmul(decoder_input, self.W_Q)
        K = torch.matmul(encoder_output, self.W_K)
        V = torch.matmul(encoder_output, self.W_V)

        # Step 2: Calculate attention map
        K_T = torch.transpose(K, -2, -1)
        QK_T = torch.matmul(Q, K_T)
        
        # Step 3: Mask future positions
        length = QK_T.size(-1)
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        QK_T.masked_fill_(mask, float('-inf')) # Fill upper triangle with negative infinity

        # Step 4: Apply softmax
        attn_map = F.softmax(QK_T, dim=-1)

        # Step 5: Normalize each row by the sum
        attn_map = attn_map / attn_map.sum(dim=-1, keepdim=True)

        # Step 6: Calculate output representations
        output = torch.matmul(attn_map, V)

        return output
 
# Question 8   

class EncoderDecoderAttention(nn.Module):
    def __init__(self, n):
        super(EncoderDecoderAttention, self).__init__()
        self.n = n
        self.W_Q = nn.Parameter(torch.randn(n, n))
        self.W_K = nn.Parameter(torch.randn(n, n))
        self.W_V = nn.Parameter(torch.randn(n, n))

    def forward(self, decoder_input, encoder_output):
        # Step 1: Calculate Q, K, V
        Q = torch.matmul(decoder_input, self.W_Q)
        K = torch.matmul(encoder_output, self.W_K)
        V = torch.matmul(encoder_output, self.W_V)

        # Step 2: Calculate attention map
        K_T = torch.transpose(K, -2, -1)
        QK_T = torch.matmul(Q, K_T)
        
        # Step 3: Mask future positions
        length = QK_T.size(-1)
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        QK_T.masked_fill_(mask, float('-inf')) # Fill upper triangle with negative infinity

        # Step 4: Apply softmax
        attn_map = F.softmax(QK_T, dim=-1)

        # Step 5: Normalize each row by the sum
        attn_map = attn_map / attn_map.sum(dim=-1, keepdim=True)

        # Step 6: Calculate output representations
        output = torch.matmul(attn_map, V)

        return output, attn_map  # Return both output and attention map
    
# Question 9

# a
'''
The self-attention matrix reflects the attention of each word in the sequence to every other word, including future words. In the encoder-decoder attention matrix,
the upper triangle is set to zero, which means that each word does not pay attention to the words that come after it in the sequence. This is known as masking future
positions. It's crucial in sequence generation tasks, where the model should not have access to future information.
'''

# b
# i
'''
In the self-attention mechanism, the representation of "yesterday" would be influenced by every other word in the sentence, including "university" or "banana
calculator". The exact impact depends on the learned weights. In the encoder-decoder attention, the representation of "yesterday" would also be affected by all words
in the sentence, but with the distinction that "yesterday" in the decoder wouldn't attend to future words. However, it's important to note that in typical
encoder-decoder models, "yesterday" in the encoder attends to all words, including future ones.
'''

# ii
'''
Comparing these attention mechanisms with human behavior, humans also pay differential attention to different parts of a sentence when understanding or generating
language. However, the distribution of attention in humans is influenced by a complex interplay of semantic, syntactic, and other factors, which may not be fully
captured by these simple attention mechanisms.
'''

# c
'''
After multiplying the attention map with V, we get a set of vectors which can be thought of as a weighted sum or average of the original value vectors (in V), where
the weights are determined by the attention map. These weighted sums are a kind of summary of the original sequence, where each summary vector gives more importance
to some parts of the sequence than to others. In the context of natural language processing, this might mean that the representation of each word in the sequence is
now influenced more by some words and less by others, depending on their attention weights.
'''

# d
'''
A model trained to predict the next word in a sequence (like a language model) could suffer from an "information leakage" problem if it uses full self-attention.
This is because each word in the sequence would have access to future words during training, which would not be the case during actual prediction. This can lead to
over-optimistic performance estimates during training and poor generalization to unseen data.

Encoder-decoder attention can be a solution to this problem by masking future positions during the decoding phase. This ensures that the prediction for each word only
depends on the previous words, just like during actual prediction. It can also enhance the model's ability to generate coherent and contextually appropriate sequences,
by focusing on the most relevant parts of the input sequence.
'''

# Question 10

class MultiHeadAttention(nn.Module):
    def __init__(self, n, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.n = n  # input size
        self.d_k = n // num_heads  # dimension of keys/queries/values per head
        
        # Parameter matrices for queries, keys, and values
        self.W_Q = nn.Parameter(torch.randn(num_heads, n, self.d_k))
        self.W_K = nn.Parameter(torch.randn(num_heads, n, self.d_k))
        self.W_V = nn.Parameter(torch.randn(num_heads, n, self.d_k))
        
        # Linear transformation for the output
        self.fc = nn.Linear(n, n)
        
    def forward(self, x):
        # Generate queries, keys, values for all heads
        Q = torch.einsum("ijk,lkm->iljm", x, self.W_Q)
        K = torch.einsum("ijk,lkm->iljm", x, self.W_K)
        V = torch.einsum("ijk,lkm->iljm", x, self.W_V)
        
        # Calculate the attention scores
        scores = torch.einsum("iljk,ilkk->ilj", Q, K) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        
        # Compute the weighted average of the values
        output = torch.einsum("ilj,iljk->ilk", attn, V)
        
        # Concatenate multiple heads
        output = output.reshape(*output.shape[:-2], -1)
        
        # Apply final linear transformation
        output = self.fc(output)
        
        return output

# Question 11

'''
Analyzing the representations produced by individual attention heads and the final combined representation in a multi-head attention model can both yield interesting
insights, but for different reasons:

Individual Attention Heads: Looking at the representations produced by each individual head can help us understand what kind of information each head is focusing on
or learning to attend to. For instance, some heads might specialize in attending to syntactic relationships (like attending to the subject of a sentence when
processing a verb), while others might focus more on semantic relationships or positional information. Investigating individual attention heads can thus provide
valuable insights into the diverse strategies the model uses to process its inputs and might offer clues to the model's successes and failures.

Final Representation: On the other hand, examining the final combined representation after multi-head attention is beneficial for understanding the overall result of
the attention process. This representation is what gets passed onto the subsequent layers in the model, and it's what ultimately determines the model's output. As
such, understanding this representation can help us understand how the model makes its final decisions or predictions.

In practice, a thorough analysis might involve looking at both types of representations. It can be particularly interesting to see how the different types of
information highlighted by different heads get combined in the final representation. At the same time, it's also important to remember that the "interpretability" of
attention mechanisms is a subject of ongoing research and debate, and that visualizations of attention weights might not always correspond directly to how humans
would interpret the same data.
'''

# Analyze NLP models representations

# Question 1

from transformers import BertModel, GPT2Model, CLIPModel, GPT2LMHeadModel, BertTokenizer, GPT2Tokenizer, CLIPTextTokenizer

# Load pre-trained models
bert = BertModel.from_pretrained('bert-base-uncased')
gpt2 = GPT2Model.from_pretrained('gpt2')
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
sgpt = GPT2LMHeadModel.from_pretrained('gpt2')

# Load tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
clip_tokenizer = CLIPTextTokenizer.from_pretrained('openai/clip-vit-base-patch32')

# Question 2

sentences = [
    # Semantically similar
    ("A most profoundly pleasant, kind, and generous person", "A nice person"),
    ("The dog is chasing the cat", "The hound is pursuing the feline"),
    
    # Semantically dissimilar
    ("She always arrives early to work", "He never forgets to feed the dogs"),
    ("I love eating strawberries in summer", "Winter nights are perfect for reading books"),
    
    # Syntactically similar
    ("A very good person", "A very bad person"),
    ("She is going to the market", "He is going to the park"),
    
    # Syntactically dissimilar
    ("I love to play basketball", "Basketball is a game I enjoy"),
    ("This is a lovely day", "Lovely, this day is"),
]

# Question 3

# Function to encode sentences
def encode_sentences(model, tokenizer, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state

# Encode sentences with each model
clip_embeddings = encode_sentences(clip, clip_tokenizer, sentences)
sgpt_embeddings = encode_sentences(sgpt, sgpt_tokenizer, sentences)
gpt2_embeddings = encode_sentences(gpt2, gpt2_tokenizer, sentences)
bert_embeddings = encode_sentences(bert, bert_tokenizer, sentences)

# Get specific embeddings
clip_output = clip_embeddings
sgpt_output = sgpt_embeddings

# Get the last token's embeddings from GPT-2 outputs
gpt2_output = gpt2_embeddings[:, -1, :]

# Get the first token's embeddings from BERT outputs
bert_output = bert_embeddings[:, 0, :]

# a
'''
For CLIP and SGPT, you would take the whole output because these models generate a representation for each input sentence as a whole, considering all the words in the
sentence.
'''

# b
'''
For GPT-2, you take the last token because GPT-2 is an autoregressive model that generates sentences from left to right. It uses causal attention mechanism, so each
token can only attend to previous tokens in the sentence. The last token thus has potentially been influenced by all previous tokens in the sentence, making it a rich
and consolidated source of information about the whole sentence.
'''

# c
'''
For BERT, you typically take the first token's embedding (the "[CLS]" token in BERT's case), because BERT uses bidirectional attention and is trained in such a way
that the first token (the "[CLS]" token) is expected to hold a representation of the entire sentence's meaning for classification tasks.
'''

# Question 4

# Calculate and plot RDM
def plot_rdm(embeddings, title):
    rdm = cosine_distances(embeddings)
    plt.figure(figsize=(10,8))
    plt.imshow(rdm, cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Compute RDMs for each model's output
plot_rdm(clip_output, 'CLIP RDM')
plot_rdm(sgpt_output, 'SGPT RDM')
plot_rdm(gpt2_output, 'GPT-2 RDM')
plot_rdm(bert_output, 'BERT RDM')

# Question 5

# a
'''
If a model is "tricked" by syntactic similarities, it means that in the RDM heatmap, sentences that are syntactically similar (but semantically different) will have
a low dissimilarity score. This suggests that the model is assigning similar representations to sentences with similar syntax, irrespective of their meanings.
'''

# b
'''
Conversely, if a model is more focused on semantic similarities, semantically similar sentences will have a low dissimilarity score in the RDM, even if their syntax
is quite different. This indicates that the model is more attuned to the meaning of the sentences.
'''

# c
'''
By examining the RDMs and comparing the patterns in them, you can infer how each model is treating different aspects of the sentences. If two models show similar
patterns in their RDMs, it suggests that they are learning similar kinds of representations.
'''

# d
'''
The results should indeed be congruent with the training objectives of the models. For example, BERT is trained to understand the semantics of text (using a masked
language modeling objective, among others), so it should be more focused on semantic similarities. GPT-2, on the other hand, is trained to predict the next word in a
sentence and uses a unidirectional (left-to-right) attention mechanism, so it might be more sensitive to syntactic structures.
'''