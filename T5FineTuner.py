from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from Datapool import *
from utils import *

class T5FineTuner():
    def __init__(self, hparam):
        self.hparam = hparam
        # self.model = T5ForConditionalGeneration.from_pretrained(
        #     hparam.model_name_or_path)
        # self.model = T5ForConditionalGeneration.from_pretrained(
        #     "/mnt/workspace/ORL/saved_models_copy/strategy_FULL/model_F1_0.66977_epoch_11")
        # self.model = T5ForConditionalGeneration.from_pretrained(
        #     "/mnt/workspace/ORL/saved_models_copy/strategy_BEAM/model_F1_0.67540_iter_11_epoch_6")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "/mnt/workspace/ORL/saved_models_copy/strategy_MARGIN/model_F1_0.67388_iter_11_epoch_8")
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparam.model_name_or_path
        )
        self.configure_optimizers()
    def is_logger(self):
        return True

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._step(batch)
        return loss

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparam.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparam.learning_rate, eps=self.hparam.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None
                       ):

        # optimizer.step(closure=optimizer_closure)
        # optimizer.zero_grad()
        self.opt.step()
        self.opt.zero_grad()
        # self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(
            self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparam)
        dataloader = DataLoader(train_dataset, batch_size=self.hparam.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=2)
        t_total = (
            (len(dataloader.dataset) //
             (self.hparam.train_batch_size * max(1, self.hparam.n_gpu)))
            // self.hparam.gradient_accumulation_steps
            * float(self.hparam.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparam.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="validation", args=self.hparam)
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=2)
    def test_dataloader(self):
        val_dataset = get_dataset(
            tokenizer=self.tokenizer, type_path="test", args=self.hparam)
        return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=2)


class T5FineTunerWithAL(T5FineTuner):
    def __init__(self, hparam, strategy_name="RANDOM", warmstart_percentage=0.05):
        super().__init__(hparam)
        self.warmstart_percentage = warmstart_percentage
        self.strategy_name = strategy_name
        # self.strategy = self.strategy[strategy_name]
        self.datapool = Datapool(warmstart_percentage, super().train_dataloader())
    def resetModel(self):
        # 重置模型权重
        del self.model
        # self.model = T5ForConditionalGeneration.from_pretrained(
        #     "saved_models/model_epoch_7")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-base")
        # 重置优化器
        del self.opt
        self.configure_optimizers()
    def resetDatapool(self):
        del self.datapool
        self.datapool = Datapool(self.warmstart_percentage, super().train_dataloader())
    def update_datapool(self, strategy="RANDOM", add_percentage=0.0):
        un_data = self.datapool.getUnannotatedData()
        if strategy == "RANDOM":
            add_num = int(self.datapool.total_num * add_percentage)
            selected_idx = random.sample(range(len(self.datapool.unannotated_data)), add_num)
            # 展示得分高的 和得分低的数据
            # show_num = 10
            # print("--------------------show samples-----------------------")
            # for sample in range(show_num):
            #     print("source:",self.tokenizer.decode(un_data[sample]["source_ids"], skip_special_tokens=True))
            #     print("target:",self.tokenizer.decode(un_data[sample]["target_ids"], skip_special_tokens=True))
            self.datapool.addAnnotatedData(selected_idx)
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(device)
        cnt = 0

        un_dataloader = DataLoader(un_data, batch_size=8, shuffle=False)
            
        #output.scores是一个长度为输出的token数量的元组，元组中的每个元素为[batchsize*num_return_sequences,词表大小]的tensor
        if add_percentage > 0:
            if strategy == "BEAM":
                beam_scores=[]
                number_beams = 2
                len_penalty = 0.0
                idx = 0
                for i, batch in enumerate(un_dataloader):
                    batch = {key: value.to(device) for key, value in batch.items()} 
                    outputs = self.model.generate(
                        input_ids=batch["source_ids"],
                        attention_mask=batch["source_mask"],
                        num_beams=number_beams,
                        num_return_sequences=number_beams,
                        return_dict_in_generate=True,
                        output_scores=True,
                        length_penalty=len_penalty,
                        max_length=500
                    )
                    transition_scores = self.model.compute_transition_scores(
                        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                    )
                    # compute_transition_scores计算出来的transition_scores是一个[batchsize*num_return_sequences,输出的token数量]的tensor，相当于对于每个输出的token一个评分

                    # input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                    # 由于t5是encoder-decoder架构，所以这里input_length取1
                    generated_tokens = outputs.sequences[:, 1:] 
                    # generated_tokens是一个[batchsize,输出的token数量]的tensor，相当于每一个输出的token对应的id

                    output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
                    reconstructed_scores = transition_scores.cpu().sum(axis=1) / (output_length**len_penalty)
                    # 遍历batch中的每一条样本，计算对每一条样本的不确定程度
                    for i in range(0,int(generated_tokens.shape[0]/number_beams)):
                        beam_scores.append((idx, reconstructed_scores[i*number_beams]))
                        idx += 1
                sorted_beam_scores = sorted(beam_scores, key=lambda x: x[1])
                # print("sorted_beam_scores:", sorted_beam_scores)
                    
                selected_idx = [item[0] for item in sorted_beam_scores][:int(self.datapool.total_num * add_percentage)]
                self.datapool.addAnnotatedData(selected_idx)
            elif strategy == "MARGIN":
                beam_scores=[]
                number_beams = 2
                len_penalty = 0.0
                idx = 0
                for i, batch in enumerate(un_dataloader):
                    batch = {key: value.to(device) for key, value in batch.items()} 
                    outputs = self.model.generate(
                        input_ids=batch["source_ids"],
                        attention_mask=batch["source_mask"],
                        num_beams=number_beams,
                        num_return_sequences=number_beams,
                        return_dict_in_generate=True,
                        output_scores=True,
                        length_penalty=len_penalty,
                        max_length=500
                    )
                    transition_scores = self.model.compute_transition_scores(
                        outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=True
                    )
                    # compute_transition_scores计算出来的transition_scores是一个[batchsize*num_return_sequences,输出的token数量]的tensor，相当于对于每个输出的token一个评分

                    # input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                    # 由于t5是encoder-decoder架构，所以这里input_length取1
                    generated_tokens = outputs.sequences[:, 1:] 
                    # generated_tokens是一个[batchsize,输出的token数量]的tensor，相当于每一个输出的token对应的id

                    output_length = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
                    reconstructed_scores = transition_scores.cpu().sum(axis=1) / (output_length**len_penalty)
                    # 遍历batch中的每一条样本，计算对每一条样本的不确定程度
                    for i in range(0,int(generated_tokens.shape[0]/number_beams)):
                        beam_scores.append((idx, np.exp(reconstructed_scores[i*number_beams]) - np.exp(reconstructed_scores[i*number_beams + 1])))
                        idx += 1
                sorted_beam_scores = sorted(beam_scores, key=lambda x: x[1])
                # print("sorted_beam_scores:", sorted_beam_scores)
                    
                selected_idx = [item[0] for item in sorted_beam_scores][:int(self.datapool.total_num * add_percentage)]
                self.datapool.addAnnotatedData(selected_idx)
            elif strategy == "OTHER":
                # 假设已经知道实际的标签，根据产生的内容和实际的标签进行对比
                overall_f1_scores = []
                metric = load_metric("seqeval")
                len_penalty = 0.0
                idx = 0
                for i, batch in enumerate(un_dataloader):
                    batch = {key: value.to(device) for key, value in batch.items()} 
                    input_ids = batch['source_ids']
                    attention_mask = batch['source_mask']
                    outs = self.model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask)
                    dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).strip() for ids in outs]
                    target = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                                for ids in batch["target_ids"]]
                    texts = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                                for ids in batch["source_ids"]]
                    true_label = [generate_label(texts[i].strip(), target[i].strip()) if target[i].strip() != 'none' else [
                        "O"]*len(texts[i].strip().split()) for i in range(len(texts))]
                    pred_label = [generate_label(texts[i].strip(), dec[i].strip()) if dec[i].strip() != 'none' else [
                        "O"]*len(texts[i].strip().split()) for i in range(len(texts))]
                    for i in range(len(batch["source_ids"])):
                        overall_f1_scores.append((idx,metric.compute(predictions=[pred_label[i]], references=[true_label[i]])['overall_f1']))
                        idx += 1
                sorted_overall_f1_scores = sorted(overall_f1_scores, key=lambda x: x[1])
                print(sorted_overall_f1_scores[:10])
                selected_idx = [item[0] for item in sorted_overall_f1_scores][:int(self.datapool.total_num * add_percentage)]
                self.datapool.addAnnotatedData(selected_idx)            

        # print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
        

    # def val_dataloader(self):
    #     val_dataset = get_dataset(
    #         tokenizer=self.tokenizer, type_path="validation", args=self.hparam)
    #     return DataLoader(val_dataset, batch_size=self.hparam.eval_batch_size, num_workers=2)
