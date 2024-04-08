import requests
import io
import csv
import torch
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F


def get_sentences_3():
    url = 'https://raw.githubusercontent.com/Niazitahir/SMS_Spam_Detection_APS360/main/RawData/sms%2Bspam%2Bcollection/SMSSpamCollection'
    separated = csv.reader(io.StringIO(requests.get(url).text), delimiter='\t')
    spam, ham = [], []
    for row in separated:
        if row[1] == '': continue
        label = int(row[0] == 'spam')
        if label == 1:
            spam.append((label, row[1]))
        elif label == 0:
            ham.append((label, row[1]))

    train, valid, test = [], [], []
    for i, example in enumerate(spam):
        if i % 5 < 3:
            train.append(example)
        elif i % 5 == 3:
            valid.append(example)
        else:
            test.append(example)

    train = [example for example in train for _ in range(6)]
    for i, example in enumerate(ham):
        if i % 5 < 3:
            train.append(example)
        elif i % 5 == 3:
            valid.append(example)
        else:
            test.append(example)

    return train, valid, test


def yield_tokens(sentences):
    for _, sentence in sentences:
        yield tokenizer_c(sentence)


def get_vocab(sentences):
    vocab = build_vocab_from_iterator(yield_tokens(sentences),
                                      specials=['<UNK>', '<BOS>', '<EOS>', '<PAD>'],
                                      max_tokens=20000)
    vocab.set_default_index(vocab['<UNK>'])
    return vocab


class LSTMNet_3(nn.Module):
    def __init__(self, input_size_w, input_size_c, hidden_size_w, hidden_size_c,
                 num_classes, glove, num_layers_w=1, num_layers_c=1):
        super(LSTMNet_3, self).__init__()
        self.name = 'LSTM3'
        self.hidden_size_w = hidden_size_w
        self.hidden_size_c = hidden_size_c
        self.num_layers_w = num_layers_w
        self.num_layers_c = num_layers_c
        pretrained_embeddings = glove.vectors
        pretrained_embeddings = torch.cat((torch.zeros(4, pretrained_embeddings.shape[1]),
                                           pretrained_embeddings))
        self.emb = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.eye = torch.eye(input_size_c)
        self.rnn_w = nn.LSTM(input_size_w, hidden_size_w, num_layers_w, batch_first=True)
        self.rnn_c = nn.LSTM(input_size_c, hidden_size_c, num_layers_c, batch_first=True)
        self.fc = nn.Linear(hidden_size_w + hidden_size_c, num_classes)


    def forward(self, x, h0c0_w=None, h0c0_c=None):
        div = torch.nonzero(x[0].eq(-1))[0, 0].item()
        w = self.emb(x[:, :div])
        c = self.eye[x[:, div + 1:]]
        out_w, _ = self.rnn_w(w, h0c0_w)
        out_c, _ = self.rnn_c(c, h0c0_c)
        length_w = (torch.sum(torch.where(x[:, :div] != 3, 1, 0),
                              dim=1,
                              keepdim=True) - 1).expand(-1, self.hidden_size_w).unsqueeze(1)
        length_c = (torch.sum(torch.where(x[:, div + 1:] != 3, 1, 0),
                              dim=1,
                              keepdim=True) - 1).expand(-1, self.hidden_size_c).unsqueeze(1)
        out_w = out_w.gather(1, length_w).squeeze(1)
        out_c = out_c.gather(1, length_c).squeeze(1)
        out = torch.cat((out_w, out_c), dim=1)
        out = self.fc(out)
        return out


def get_file_path(name, glove, batch_size, hidden_size, num_layers, learning_rate):
    path = '{0}_{1}_bs{2}_hs{3}_nl{4}_lr{5}_'.format(name,
                                                     glove,
                                                     batch_size,
                                                     hidden_size,
                                                     num_layers,
                                                     learning_rate)
    return './' + path


def evaluate_sample_2(sample, model, transform_w, transform_c):
    idxs = torch.tensor(sentence_transform_w(sample) + [-1] + sentence_transform_c(sample))
    pred = model(idxs.unsqueeze(0))
    pred = F.softmax(pred, dim=1)[0, 1].item()
    if pred > 0.3:
        print('<<This message may be spam.>>')


train, valid, test = get_sentences_3()

tokenizer_w = get_tokenizer('basic_english')
tokenizer_c = get_tokenizer(lambda x: list(x))

glove_vectors = torchtext.vocab.GloVe(name='twitter.27B', dim=50)
vocab_w = torchtext.vocab.vocab(glove_vectors.stoi)
vocab_w.insert_token('<UNK>', 0)
vocab_w.insert_token('<BOS>', 1)
vocab_w.insert_token('<EOS>', 2)
vocab_w.insert_token('<PAD>', 3)
vocab_w.set_default_index(0)
vocab_c = get_vocab(train)

sentence_transform_w = lambda x: [vocab_w['<BOS>']] + [vocab_w[token] for token in
                                                       tokenizer_w(x)] + [vocab_w['<EOS>']]
sentence_transform_c = lambda x: [vocab_c['<BOS>']] + [vocab_c[token] for token in
                                                       tokenizer_c(x)] + [vocab_c['<EOS>']]

tuned_model = LSTMNet_3(50, len(vocab_c), 25, len(vocab_c) // 2,
                  2, glove_vectors, num_layers_w=2, num_layers_c=2)
path = get_file_path('LSTM3', 'twi50', 16, 25, 2, 2e-4)
tuned_model.load_state_dict(torch.load(path + '23', map_location=torch.device('cpu')))

print('\nMessage Inbox:')
while(1):
    evaluate_sample_2(input(), tuned_model, sentence_transform_w, sentence_transform_c)
