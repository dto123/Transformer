# import necessary modules
import torch
import torch.nn as nn
import sys
import os
sys.path.append('OpenNMT-py')
import time
import onmt
import onmt.inputters
import onmt.modules
import onmt.utils

# build vocab

vocab_fields = torch.load("data.vocab.pt")

src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

# define model and loss function
num_layers = 4 #number of encoder/decoder layers
d_model = 256 #size of the model
heads = 2 #number of heads
d_ff = 1024 #size of the inner FF layer
dropout = 0.3 #dropout parameters
copy_attn = False
self_attn_type = "scaled-dot"
max_relative_positions=0

batch_size = 128
emb_size = 256

# Specify the core model.

encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab),
                                             word_padding_idx=src_padding,
                                             position_encoding=True)

encoder = onmt.encoders.TransformerEncoder(num_layers=num_layers, d_model=d_model, 
                                           heads=heads, d_ff=d_ff, dropout=dropout,
                                           embeddings=encoder_embeddings,
                                           max_relative_positions=max_relative_positions)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab),
                                             word_padding_idx=tgt_padding,
                                             position_encoding=True)

decoder = onmt.decoders.TransformerDecoder(num_layers=num_layers, d_model=d_model, 
                                           heads=heads, d_ff=d_ff, copy_attn=copy_attn, 
                                           self_attn_type=self_attn_type, dropout=dropout, 
                                           embeddings=decoder_embeddings,
                                           max_relative_positions=max_relative_positions)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = onmt.models.model.NMTModel(encoder, decoder)
model.to(device)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
    nn.Linear(d_model, len(tgt_vocab)),
    nn.LogSoftmax(dim=-1)).to(device)

loss = onmt.utils.loss.NMTLossCompute(
    criterion=nn.CrossEntropyLoss(ignore_index=tgt_padding, reduction="sum"),
    generator=model.generator)


checkpoint_path=None
checkpoint_path='_step_8000.pt' # put checkpoint here

# define optimizer

class Opt:
  decay_method = 'noam'
  warmup_steps = 8000
  rnn_size = 512
  learning_rate=2
  max_grad_norm=0
  adam_beta1 = 0.9
  adam_beta2 = 0.998
  model_dtype='fp32'
  optim='adam'
  batch_size=batch_size
  valid_batch_size=batch_size
  batch_type='tokens'
  max_generator_batches=2
  warmup_steps=8000
  param_init=0
  param_init_glorot=True
  label_smoothing=0.1
  train_from=checkpoint_path
  reset_optim = 'none'
  
opt=Opt()

if checkpoint_path is not None:
  checkpoint=torch.load(checkpoint_path)
  checkpoint['model']['generator.0.weight']=checkpoint['generator']['0.weight']
  checkpoint['model']['generator.0.bias']=checkpoint['generator']['0.bias']
  model.load_state_dict(checkpoint['model'])
else:
  checkpoint=None
  
optim = onmt.utils.optimizers.Optimizer.from_opt(model, opt, checkpoint)

#lr = 2
#betas=(0.9, 0.998)

#torch_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
#optim = onmt.utils.optimizers.Optimizer(torch_optimizer, learning_rate=lr, max_grad_norm=0)


# load our data
from itertools import chain
train_data_file = "data.train.0.pt"
valid_data_file = "data.valid.0.pt"


train_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[train_data_file],
                                                     fields=vocab_fields,
                                                     batch_size=batch_size,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=True,
                                                     repeat=True)

valid_iter = onmt.inputters.inputter.DatasetLazyIter(dataset_paths=[valid_data_file],
                                                     fields=vocab_fields,
                                                     batch_size=batch_size,
                                                     batch_size_multiple=1,
                                                     batch_size_fn=None,
                                                     device=device,
                                                     is_train=False,
                                                     repeat=False)


class ColabModelSaver(onmt.models.model_saver.ModelSaverBase):
    """Simple model saver to filesystem"""

    def _save(self, step, model):
        real_model = (model.module
                      if isinstance(model, nn.DataParallel)
                      else model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': self.fields,
            'opt': self.model_opt,
            'optim': self.optim.state_dict(),
        }

        print("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        torch.save(checkpoint, checkpoint_path)
        
        uploaded = drive.CreateFile({'title': checkpoint_path})
        uploaded.SetContentFile(checkpoint_path)
        uploaded.Upload()
        print('Uploaded file with ID {}'.format(uploaded.get('id')))
        os.remove(checkpoint_path)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)
        
          
class ColabReportMgr(onmt.utils.report_manager.ReportMgrBase):

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        Done every report_every steps
        """
        t = report_stats.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        print(
            ("Step %s; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " +
             "lr: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step_fmt,
               report_stats.accuracy(),
               report_stats.ppl(),
               report_stats.xent(),
               learning_rate,
               report_stats.n_src_words / (t + 1e-5),
               report_stats.n_words / (t + 1e-5),
               time.time() - self.start_time))
        
        report_stats = onmt.utils.Statistics()
        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        Done everytime we do validation
        """
        if train_stats is not None:
            print('Train perplexity: %g' % train_stats.ppl())
            print('Train accuracy: %g' % train_stats.accuracy())

        if valid_stats is not None:
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())




# train the model
report_manager = ColabReportMgr(report_every=100)
model_saver=ColabModelSaver('',model,opt,vocab_fields,optim)


trainer = onmt.Trainer(model=model,
                       train_loss=loss,
                       valid_loss=loss,
                       optim=optim,
                       report_manager=report_manager,
                       model_saver=model_saver,
                       norm_method='tokens',
                       accum_count=[2],
                       )

trainer.train(train_iter=train_iter,
              train_steps=200000,
              save_checkpoint_steps=1000,
              valid_iter=valid_iter,
              valid_steps=1000)


# translate the validation set

import onmt.translate

src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = onmt.translate.GNMTGlobalScorer(alpha=0.7, 
                                         beta=0., 
                                         length_penalty="avg", 
                                         coverage_penalty="none")
gpu = 0 if torch.cuda.is_available() else -1
translator = onmt.translate.Translator(model=model, 
                                       fields=vocab_fields, 
                                       src_reader=src_reader, 
                                       tgt_reader=tgt_reader, 
                                       global_scorer=scorer,
                                       gpu=gpu)
builder = onmt.translate.TranslationBuilder(data=torch.load(valid_data_file), 
                                            fields=vocab_fields)

with open('output_dev.en', 'w') as out:
  for batch in valid_iter:
      trans_batch = translator.translate_batch(
          batch=batch, src_vocabs=[src_vocab],
          attn_debug=False)
      translations = builder.from_batch(trans_batch)
      for trans in translations:
          out.write(' '.join(trans.pred_sents[0]) + '\n')
          print(trans.log(0))              



