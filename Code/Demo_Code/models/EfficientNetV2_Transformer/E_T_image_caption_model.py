import torch
import torch.nn as nn
import torchvision
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Position Encoding class
class position_encoding(nn.Module):
    def __init__(self, d_model=512, max_len=35, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, ::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(0) > self.pe.size(0):
            self.pe = self.pe.repeat(x.size(0), 1, 1)
        self.pe = self.pe[:x.size(0), :, :]
        return self.dropout(self.pe + x)

# Image Captioning Model class
class Imagecaptionmodel(nn.Module):
  def __init__(self, vocab_size=1823, embedding_size=1280, max_len=35, n_head=16, num_decoder_layer=4):
    super().__init__()
    self.position_encoding = position_encoding(d_model = embedding_size)

    self.transformer_decoder_layer  = nn.TransformerDecoderLayer(d_model = embedding_size, nhead = n_head)
    self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers = num_decoder_layer)
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.FC = nn.LazyLinear(vocab_size)
    self.initweights()
    self.embedding_size = embedding_size
  def initweights(self):
    self.embedding.weight.data.uniform_(-0.1, 0.1)
    self.FC.weight.data.uniform_(-0.1, 0.1)
    self.FC.bias.data.zero_()
  def create_mask(self, seq):
      'create mask for mask attention'
      attention_mask  = torch.ones(seq.size(1), seq.size(1))
      # print('attention_mask ',attention_mask.shape)
      attention_mask  = torch.tril(attention_mask)
      attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf')).masked_fill(attention_mask == 1, 0)

      pad_mask = seq.masked_fill(seq == 0, float(0.0)).masked_fill(seq > 0, float(1.0))
      pad_mask_bool = seq == 0
      return attention_mask, pad_mask, pad_mask_bool
  def forward(self,seq, image_embedding):
    image_embedding  = image_embedding.permute(1,0,2) # 49,32,512
    # print(image_embedding)
    # print('create_mask ')
    # print('seq', seq.shape)
    x = self.embedding(seq) * math.sqrt(self.embedding_size)
    x = self.position_encoding(x) # 32, 33 512
    x = x.permute(1, 0, 2) # (seqlen, batchsize, embedding)
    # print('x permute', x.shape)
    # print('image_embedding shape ', image_embedding.shape)
    attention_mask, pad_mask, pad_mask_bool = self.create_mask(seq)
    attention_mask, pad_mask, pad_mask_bool = attention_mask.to(device), pad_mask.to(device), pad_mask_bool.to(device)
    # print('done embedding x shape ',x.shape )
    # print('attention_mask, pad_mask, pad_mask_bool', attention_mask.shape, pad_mask.shape, pad_mask_bool.shape)

    #model nhan vao tgt: (seq_len, batch_size, embddingsize)
    #memory (input_seq_len, batch_size, embeddingsize)
    # tgt_key_padding_mask = (N, T)
    # tgt_mask = (T, T)
    # memory_mask: (T, S)
    x = self.transformer_decoder(memory = image_embedding, tgt = x, tgt_mask = attention_mask, tgt_key_padding_mask = pad_mask_bool
                                 ) #(T, N, E) (33, 32, 512)
    # print('out transformer :',x.shape)
    out = self.FC(x)
    return out, pad_mask
