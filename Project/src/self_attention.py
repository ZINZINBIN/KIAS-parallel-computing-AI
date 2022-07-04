import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class SelfAttention(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length):
		super(SelfAttention, self).__init__()
  
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length

		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.dropout = 0.5
		self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=self.dropout, bidirectional=True)
		self.W_s1 = nn.Linear(2*hidden_size, 256)
		self.W_s2 = nn.Linear(256, 32)
		self.fc_layer = nn.Linear(32*2*hidden_size, 512)
		self.label = nn.Linear(512, output_size)

	def attention_net(self, lstm_output):

		attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
		attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
		attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

		return attn_weight_matrix

	def forward(self, input_sentences, batch_size=None):

		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		if batch_size is None:
			h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())
		else:
			h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())

		output, (h_n, c_n) = self.bilstm(input, (h_0, c_0))
		output = output.permute(1, 0, 2)
		attn_weight_matrix = self.attention_net(output)

		hidden_matrix = torch.bmm(attn_weight_matrix, output)
		fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2]))
		logits = self.label(fc_out)

		return logits