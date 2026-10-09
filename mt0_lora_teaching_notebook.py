#!/usr/bin/env python
# coding: utf-8

# # Multilingual News: Topic Classification + Headline Generation with mT0 + LoRA
# 
# **Goal.** Given a BBC news article in **Hausa, Yoruba, Igbo or Nigerian Pidgin**, predict two things:
# 1. its **topic** (e.g. `sports`, `politics`, `health`) — a classification task
# 2. a **headline** — a text-generation task
# 
# **The idea in one sentence.** We fine-tune *one* multilingual text-to-text model (`bigscience/mt0-base`) to do *both* jobs, telling it which job to do with a short prefix (`classify topic:` / `headline topic:`), and we train it cheaply with **LoRA** so it fits on a free Kaggle GPU.
# 
# **Roadmap**
# 1. Setup & imports
# 2. Explore the data (EDA)
# 3. Prepare the data → clean, dedupe, check leakage, split, write JSONL
# 4. Fine-tune mT0 with LoRA
# 5. Save the adapter
# 6. Load it back and run inference (classification by label-scoring, headlines by beam search)

# ## 1. Setup
# Import libraries, fix a package version, and locate the competition files.

# In[1]:


# =============================================================================
# STEP 0 — Imports and a look at what files Kaggle has mounted for us
# =============================================================================
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# --- Classic data-science toolkit -------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   # quick charts to sanity-check the data
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Walk every folder under /kaggle/input and print each file path.
# This is the fastest way to confirm WHERE your competition data and any
# uploaded models live, before you hard-code paths later in the notebook.
# (Notice the output shows both the competition CSVs AND your own uploaded
#  LoRA adapter under /kaggle/input/models/... — we'll load that at the end.)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub                     # downloads/attaches competition data & models
import unicodedata                   # Unicode normalisation (vital for Yoruba/Igbo tone marks)
import re                            # regular expressions for whitespace cleanup
import json                          # reading/writing JSON + JSONL files
from sklearn.model_selection import train_test_split   # stratified train/validation split
import torch                         # the deep-learning engine underneath Hugging Face
from datasets import load_dataset    # Hugging Face `datasets`: loads our JSONL files efficiently
import os                            # (duplicate import — harmless, Python just reuses the module)
# transformers gives us the pretrained model + the high-level training loop:
#   AutoTokenizer            -> turns text into token IDs (and back)
#   AutoModelForSeq2SeqLM    -> an encoder-decoder model (text in -> text out)
#   Seq2SeqTrainer           -> the training loop (batching, eval, checkpoints...)
#   DataCollatorForSeq2Seq   -> pads each batch to the same length
#   Seq2SeqTrainingArguments -> all the training hyper-parameters in one object
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,Seq2SeqTrainer,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments
# peft = Parameter-Efficient Fine-Tuning. Instead of updating all ~580M weights
# we bolt small trainable "LoRA" matrices onto the frozen model (~1% of params).
from peft import LoraConfig, TaskType, get_peft_model
from peft import PeftModel           # used later to re-load a saved LoRA adapter for inference
import json                          # (duplicate import — harmless)
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# In[2]:


# =============================================================================
# Fix a library version conflict
# =============================================================================
# The `!` prefix runs a shell command from inside the notebook.
# Kaggle's image ships torchao 0.10, but the newer transformers/peft versions
# expect torchao >= 0.16, and importing them can fail with the old one.
# So we uninstall the old copy and install a compatible version.
# Tip: if imports still complain after this, restart the kernel so Python
# picks up the freshly installed package.
!pip uninstall -y torchao
!pip install -U "torchao>=0.16.0"

# In[3]:


# Download latest version
# competition_download() returns the local folder holding train.csv / test.csv.
# On Kaggle the data is already mounted, so this just returns the path instantly.
path = kagglehub.competition_download('dsn-bootcamp-hackathon-2026-llm-agent-track')

# The base model we will fine-tune:
#   mT0 = mT5 (a multilingual T5 encoder-decoder trained on 101 languages)
#         + instruction tuning on many prompted tasks.
# "base" is ~580M parameters — small enough to train on a free Kaggle GPU.
# Being multilingual matters here: our data is Hausa, Yoruba, Igbo and
# Nigerian Pidgin, which English-only models handle poorly.
MODEL_NAME='bigscience/mt0-base'
print("Path to competition files:", path)

# In[4]:


# Load the raw CSVs into pandas DataFrames.
#   train.csv -> has text + the answers we learn from (category, headline)
#   test.csv  -> has only id, lang and text; we must PREDICT category & headline
df_train=pd.read_csv(path+"/train.csv")
df_test=pd.read_csv(path+"/test.csv")

# ## 2. Explore the data (EDA)
# Before modelling, look at the data: what columns exist, how many rows, and how examples are spread across languages. This tells you whether the dataset is balanced and what a sensible validation split looks like.

# In[5]:


# Always eyeball a few rows first.
# Columns: category (topic label), headline (target for generation),
# text (the article body = model input), url, id, split, lang (hau/yor/ibo/pcm).
df_train.head()

# In[6]:


# =============================================================================
# EDA — how many articles do we have per language? (TRAIN set)
# =============================================================================
# Why look? If one language dominates, the model may be weaker on the others,
# and we want our validation split to keep the same mix (see stratify() later).
# Result: Hausa ~2.2k, Yoruba ~1.4k, Igbo ~1.4k, Pidgin ~1.1k -> imbalanced but OK.
#
# Note: this cell counts LANGUAGES, even though variable names / chart labels
# say "category". The topic labels live in the `category` column — try
# df_train['category'].value_counts() as well to see the class balance.
print("Total training datasets:",len(df_train['lang']))
list_of_categories = df_train['lang'].value_counts()   # count rows per language
print("Unique list of categories:",list_of_categories)
list_of_categories.plot(kind='bar', color='skyblue', edgecolor='black')
# Formatting
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Keeps category letters upright
plt.tight_layout()
plt.show()

# In[7]:


# Same check for the TEST set.
# The test language mix (hau 637, yor 411, ibo 390, pcm 305) mirrors training,
# which is good news: the model is evaluated on a similar distribution.
print("Total training datasets:",len(df_test['lang']))
list_of_categories = df_test['lang'].value_counts()
print("Unique list of categories:",list_of_categories)
list_of_categories.plot(kind='bar', color='skyblue', edgecolor='black')
# Formatting
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Keeps category letters upright
plt.tight_layout()
plt.show()

# In[49]:


# Test rows: only id, lang, text. No category/headline — that's what we predict.
df_test.head()

# ## 3. Data preparation
# 
# Models are only as good as their input. `PrepareDataset` does six things:
# 
# | Step | Why it matters |
# |---|---|
# | Unicode NFC + whitespace cleanup | Yoruba/Igbo tone marks can be encoded in several ways; the tokenizer should see one consistent form |
# | Lowercase labels | `Sports` and `sports` must be the same class |
# | Deduplicate | Repeated articles skew training and inflate validation scores |
# | Leakage check | An article in both train and test lets the model "cheat" |
# | Stratified split | Validation keeps the same mix of topic × language as training |
# | Write JSONL with task prefixes | Each article becomes **two** text-to-text examples — one per task |
# 
# Example of the two rows produced from one article:
# ```json
# {"task": "classify", "text": "classify topic: <article>", "tgt": "sports"}
# {"task": "headline", "text": "headline topic: <article>", "tgt": "Muller ne aka zaba ..."}
# ```

# In[8]:


class PrepareDataset:
    # Turns the raw CSVs into clean JSONL files the trainer can read.
    #
    # Pipeline (see process()):
    # 1. clean text       -> consistent Unicode + whitespace
    # 2. lowercase labels -> 'Sports' and 'sports' become one class
    # 3. deduplicate      -> no repeated articles
    # 4. leak check       -> no article appears in both train and test
    # 5. stratified split -> 80% train / 20% validation
    # 6. write JSONL      -> one example per line, per task
    #

    # ONE model learns TWO tasks. The text prefix tells it which job to do —
    # this is the classic T5 "text-to-text" trick: every task is input text ->
    # output text, and the prefix is the instruction.
    #   "classify topic: <article>"  -> "sports"
    #   "headline topic: <article>"  -> "Muller ne aka zaba ..."
    # IMPORTANT: inference must use EXACTLY the same prefixes (see note in the
    # evaluation section — the headline prefix there currently differs).
    PREFIX={"classify":'classify topic: ','headline':'headline topic: '}
    OUT_FILE='/kaggle/working'   # Kaggle's writable output folder

    def clean(self, text):
        # Guard: NaN / None / non-strings become an empty string so later code never crashes.
        if  not text or not isinstance(text, str):
            return ""
        # NFC normalisation: the letter "ẹ́" can be stored as one code point or as
        # "e" + dot-below + acute accent. They LOOK identical but tokenize
        # differently. NFC picks one canonical form so the tokenizer sees
        # consistent input — very important for tonal Yoruba and Igbo.
        s = unicodedata.normalize("NFC", text)
        # Remove invisible characters that sneak in from web scraping:
        #   ​ zero-width space, ﻿ byte-order mark, \xa0 non-breaking space.
        # (Small caveat: replacing \xa0 with "" can glue two words together;
        #  replacing it with " " would be safer.)
        s = s.replace("​","").replace("﻿","").replace("\xa0", "")
        # Collapse runs of spaces/tabs/newlines into a single space, trim the ends.
        s = re.sub(r"\s+"," ",s).strip()
        return s

    def data_leak(self, df_train,df_test):
        # "Data leakage" = the same example in train and test. The model would just
        # memorise it, making scores look better than reality.
        # Set intersection (&) finds article texts present in BOTH sets.
        overlap = set(df_train['text']) & set(df_test['text'])
        if overlap:
            print("Overlap present:", len(overlap))
            # Here we DROP the overlapping rows from TEST.
            # ⚠️ For a Kaggle submission this is risky: every test id usually needs
            # a prediction, and these 29 ids would be missing. The usual fix is to
            # drop the overlap from TRAIN instead and keep the test set intact.
            df_test=df_test[~df_test['text'].isin(overlap)].reset_index(drop=True)
        else:
            print("No overlap")
        return df_train, df_test

    def stratify(self, df):
        # Hold out 20% as a validation set to measure progress during training.
        # stratify=category|lang (e.g. "sports|hau") keeps the proportion of every
        # (topic, language) pair the same in both splits, so validation isn't
        # accidentally missing, say, Igbo religion articles.
        # random_state=42 makes the split reproducible run after run.
        train, val = train_test_split(df, train_size=0.80, stratify=df['category']+'|'+df['lang'],random_state=42)
        return train, val

    def store_output(self, df,file_name):
        # JSONL = one JSON object per line. Hugging Face `datasets` reads it
        # directly and it streams well for large files.
        # ensure_ascii=False keeps "ẹ" as-is instead of "ẹ" (readable + smaller).
        with open(os.path.join(self.OUT_FILE,f"{file_name}.jsonl"),"w",encoding='utf-8') as f:
            for r in df.itertuples():          # itertuples() is much faster than iterrows()
                rid = getattr(r, "id")
                # Each article becomes TWO training examples — one per task.
                # That's why train.jsonl ends up with ~2x the number of articles.
                if hasattr(r,'category'):
                    f.write(json.dumps({'id':rid,'task':'classify','text':self.PREFIX['classify'] + r.text,'lang':r.lang,'tgt':r.category},ensure_ascii=False)+"\n")
                if hasattr(r,'headline'):
                    f.write(json.dumps({'id':rid,'task':'headline','text':self.PREFIX['headline']+r.text,'lang':r.lang,'tgt':r.headline},ensure_ascii=False)+"\n")
                # Note: test rows have neither 'category' nor 'headline', so
                # test.jsonl is written EMPTY. That's fine because inference
                # below reads df_test directly — just don't rely on test.jsonl.

    def process(self, df_train, df_test):
        # Which columns to clean in each DataFrame.
        train_cols = df_train[['headline','text','lang','category']].columns
        test_cols = df_test[['text','lang']].columns
        #Cleaning
        def clean_data(df, cols):
            for c in cols:
                if c not in {'category'}:      # labels are short; we only lowercase them below
                    df[c] = df[c].map(self.clean)
            return df
        df_train = clean_data(df_train, train_cols)
        df_test = clean_data(df_test, test_cols)

        #changing to the lower case if any
        # Normalising case prevents "Sports" and "sports" from becoming two classes.
        df_train['lang'] = df_train['lang'].str.lower()
        df_train['category'] = df_train['category'].str.lower()
        df_test['lang'] = df_test['lang'].str.lower()

        #Deduplication and data leak
        # Duplicate articles over-weight some examples and can leak between
        # train/val after splitting. 6068 -> 6003 rows (65 duplicates removed).
        print("Train: Before dropping: ", len(df_train))
        df_train = df_train.drop_duplicates(subset=['text'])
        print("Train: After dropping: ", len(df_train))
        print("Test: Before dropping: ", len(df_test))
        # ⚠️ This line does nothing: drop_duplicates() RETURNS a new DataFrame,
        # and the result isn't assigned. (For test that's arguably what we want —
        # every test id needs a prediction — but then the line can just go.)
        df_test.drop_duplicates(subset=['text'])
        print("Test: Before dropping: ", len(df_test))
        df_train, df_test = self.data_leak(df_train,df_test)
        print("After data_leak:",len(df_test))
        train,val = self.stratify(df_train)
        #Storing the files
        self.store_output(train,'train')
        self.store_output(val,'val')
        self.store_output(df_test,'test')
        print(df_train.columns)
        # Save the sorted list of class names. The classifier at inference time
        # scores exactly these strings, so this file is the "label vocabulary".
        # (It's a single JSON array, not JSON-Lines, despite the .jsonl name.)
        with open(os.path.join(self.OUT_FILE,'label.jsonl'),"w",) as l:
            json.dump(sorted(df_train["category"].unique()), l, ensure_ascii=False)
        return df_train,df_test

        

# In[9]:


# Run the whole preparation pipeline.
# Output to read: 65 duplicate training articles dropped, 29 test articles also
# found in train (leak) and removed from test -> 1713 test rows remain.
prepare_dataset_obj = PrepareDataset()
df_t, df_te = prepare_dataset_obj.process(df_train,df_test)

# ## 4. Fine-tuning with LoRA
# 
# **Why mT0?** It's an encoder–decoder (T5-style) model pretrained on 100+ languages and then instruction-tuned, so it already understands "do X with this text" prompts. Text-in → text-out means both of our tasks use the same model and the same loss.
# 
# **Why LoRA?** Full fine-tuning updates all ~580M weights (lots of GPU memory, a 2 GB checkpoint per run). LoRA freezes the original weights and learns a small low-rank correction for each linear layer:
# 
# $$W' = W + \frac{\alpha}{r} \, B A, \qquad A \in \mathbb{R}^{r\times d},\; B \in \mathbb{R}^{d\times r},\; r \ll d$$
# 
# Only `A` and `B` are trained → **~6.8M trainable parameters (≈1.15%)**, and the saved adapter is just a few MB.
# 
# **Memory tricks used:** small batches + gradient accumulation, gradient checkpointing, and bf16 where the GPU supports it.
# 
# **Reading the step count:** with 2 × T4 GPUs, effective batch = 2 (per device) × 2 (accumulation) × 2 (GPUs) = 8. 9,604 training rows ÷ 8 ≈ 1,201 steps/epoch × 5 epochs = **6,005 steps**.

# In[10]:


class NewsModelTrainer:
    # Loads the JSONL files, tokenizes them and fine-tunes mT0 with LoRA.

    def __init__(self, model_name, out_location,resume):
        self.out_location = out_location
        # The tokenizer MUST match the model: it maps text pieces ("sub-words")
        # to the integer IDs the model was pretrained on.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        # 'both' = keep classify AND headline examples -> multi-task training.
        train_ds = self._load_datasets(out_location,'train','both')
        val_ds = self._load_datasets(out_location,'val','both')
        # .map(batched=True) tokenizes many rows at once (fast).
        # remove_columns drops the raw text/id/lang columns, leaving only what
        # the model consumes: input_ids, attention_mask, labels.
        # 9604 train rows = 4802 articles x 2 tasks; 2402 val rows = 1201 x 2.
        self.train=train_ds.map(self._tokenize, batched=True, remove_columns=train_ds.column_names)
        self.val=val_ds.map(self._tokenize, batched=True, remove_columns=train_ds.column_names)
        self.resume = resume
    def _load_datasets(self,data_dir, split, task):
        # load_dataset("json") reads JSONL. With a single file HF calls it the
        # "train" split internally — hence split="train" even for val.jsonl.
        ds = load_dataset("json",data_files=os.path.join(data_dir, f"{split}.jsonl"),split="train")
        # Optionally keep only one task (handy for training single-task models).
        return ds if task=='both' else ds.filter(lambda e:e['task'] == task)

    def _tokenize(self, ds):
        # Encoder input: the article (with its task prefix), cut at 512 tokens.
        # Longer articles lose their tail — fine, the lead paragraph carries the topic.
        x = self.tokenizer(ds['text'], max_length=512, truncation=True)
        # Decoder target ("labels"): the answer text. 48 tokens is plenty for a
        # one-word category or a headline. text_target= tells the tokenizer
        # these are targets (matters for some models' special tokens).
        x['labels'] = self.tokenizer(text_target=ds['tgt'], max_length=48, truncation=True)['input_ids']
        return x

    def train_model(self):
        # Dropout randomly zeroes activations during training -> less overfitting.
        kw = {"dropout_rate": 0.1}
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kw)

        # ---------------------------------------------------------------
        # LoRA (Low-Rank Adaptation) — the key idea of this notebook
        # ---------------------------------------------------------------
        # Freeze the original weight matrix W. Learn a small update
        #     W' = W + (alpha/r) * B @ A
        # where A is (r x d) and B is (d x r) with r tiny (16) vs d (768).
        # Result: ~6.8M trainable params instead of ~589M (1.15%, see output).
        # Benefits: fits on a T4, trains fast, and the saved adapter is only a
        # few MB instead of a 2GB model copy.
        lora = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,                        # rank: 8 (small data) .. 32 (lots of data)
            lora_alpha=32,           # scaling = alpha / r; keep alpha = 2r
            lora_dropout=0.05,  # 0.05; raise to 0.1 if val loss rises early
            target_modules="all-linear",  # q,k,v,o + FFN in encoder, decoder, cross-attn
            bias="none",                 # don't train bias terms either
        )
        model = get_peft_model(model, lora)      # wraps the model, freezes the base
        model.print_trainable_parameters()       # sanity check: should be ~1%
        dev = "cuda" if torch.cuda.is_available() else "cpu"   # (not used below; Trainer picks the device itself)
        # bf16 = 16-bit "brain float": halves memory, keeps fp32's numeric range.
        # ⚠️ On a T4 GPU (your Kaggle accelerator) bf16 is NOT native — PyTorch
        # reports True because it can *emulate* it, but that's slow. On T4,
        # plain fp32 is usually faster; on A100/H100/L4 bf16 is the right choice.
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        print(bf16)
        targs = Seq2SeqTrainingArguments(
            output_dir="adapters/both_ckpt",     # where checkpoints are saved
            # Effective batch = per_device (2) x accumulation (2) x #GPUs.
            # With "GPU T4 x2" that is 2 x 2 x 2 = 8, so one epoch over 9604 rows
            # is ~1201 steps and 5 epochs = 6005 steps — exactly what the log shows.
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,       # sum gradients over 2 mini-batches before updating
            # Gradient checkpointing: don't keep every activation in memory;
            # recompute them in the backward pass. Trades ~30% speed for big
            # memory savings — needed for 512-token inputs on a 16GB T4.
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=5e-4,                  # LoRA tolerates a higher LR than full fine-tuning (~1e-4..1e-3)
            lr_scheduler_type="linear",          # LR decays linearly to 0 by the end
            num_train_epochs=5,
            warmup_ratio=0.05,                   # first 5% of steps ramp LR up from 0 (stabilises early training)
            weight_decay=0.01,                   # mild L2-style regularisation
            max_grad_norm=1.0,                   # gradient clipping: prevents exploding updates
            bf16=bf16,
            fp16=False,  # mT5-family models overflow in fp16
            eval_strategy="epoch",               # compute validation loss after each epoch
            save_strategy="epoch",               # save a checkpoint after each epoch
            save_total_limit=2,                  # keep only the 2 newest checkpoints (disk space)
            load_best_model_at_end=True,         # at the end, reload the checkpoint with lowest...
            metric_for_best_model="eval_loss",   # ...validation loss (a simple form of early stopping)
            logging_steps=50,
            report_to="none",                    # don't send logs to W&B etc.
            seed=42,                             # reproducibility
        )
        trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=self.train, eval_dataset=self.val,
        # The collator pads every batch to its longest sequence. Label padding
        # uses -100, which the loss function IGNORES, so padding never counts
        # as a "wrong answer". pad_to_multiple_of=8 helps GPU tensor cores.
        data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=model, label_pad_token_id=-100,
                                             pad_to_multiple_of=8),
        )
        print(trainer.args.device)              # cuda:0
        print(next(model.parameters()).device)  # cuda:0 once training starts
        # Resuming lets you continue after Kaggle's session timeout instead of
        # starting over: Trainer restores weights, optimizer, scheduler and step.
        if self.resume:
            ckpt_dir = "adapters/both_ckpt"
            if self.resume is True and not (os.path.isdir(ckpt_dir) and any(
                d.startswith("checkpoint-") for d in os.listdir(ckpt_dir))):
                print(f"no checkpoint in {ckpt_dir}; starting from scratch")
                # ⚠️ Bug: this sets a LOCAL variable `resume`, but the next line
                # passes self.resume (still True). With no checkpoint present,
                # trainer.train(resume_from_checkpoint=True) raises an error.
                # Fix: write `self.resume = None` here.
                resume = None
            trainer.train(resume_from_checkpoint=self.resume)
        else:
            trainer.train()
        return model, self.tokenizer    

# In[22]:


# Build the trainer: loads train/val JSONL and tokenizes them (the "Map" bars).
# resume=True -> continue from the latest checkpoint in adapters/both_ckpt if one exists.
trainer = NewsModelTrainer(MODEL_NAME, '/kaggle/working', resume= True)

# In[23]:


 # Kick off fine-tuning. Expect several hours on a T4 for 5 epochs.
 # In this saved run the bar shows 6005/6005 immediately and an empty loss table:
 # it resumed from a checkpoint that had ALREADY finished, so there was nothing left to train.
 model, tokenizer = trainer.train_model()

# ## 5. Saving the model
# We save only the LoRA adapter. To use it, you always load the base model first and then attach the adapter.

# In[25]:


# Save ONLY the LoRA adapter (a few MB), not the full base model.
# To use it later: load the base model (mt0-base) and attach this adapter on
# top — exactly what load_model() does below. Saving the tokenizer alongside
# keeps the pair together. Upload this folder as a Kaggle Model to reuse it
# across notebooks (that's where model_path below points).
model_out='/kaggle/working/adapters/both'
os.makedirs(model_out,exist_ok=True)
model.save_pretrained(model_out)
tokenizer.save_pretrained(model_out)

# ## 6. Inference & evaluation
# 
# Two different decoding strategies, one per task:
# 
# - **Classification → label scoring (not free generation).** For every allowed label we compute how likely the model is to write that exact label, then pick the most likely one. The output is always a valid class, and we get a confidence score for free.
# - **Headline → beam search.** Explore several candidate headlines in parallel and keep the best, with a rule that blocks repeated 3-word phrases.
# 
# > ⚠️ **Prefix check.** The prompts used here must be identical to the ones used in training. Classification matches (`classify topic: `), but the headline prefix here is `write headline: ` while training used `headline topic: `.

# In[56]:


# ---- Inference configuration ----
# Your trained adapter, uploaded as a Kaggle Model (so you don't retrain every session).
model_path="/kaggle/input/models/swapnilsoni/model-lora/pytorch/default/1/model_lora"
# The list of possible topics saved during data prep, e.g. ['business', 'health', ...].
lables = json.load(open('/kaggle/working/label.jsonl','r',encoding='utf-8'))
n = len(lables)   # number of classes
# ⚠️ These prefixes must match training EXACTLY. Training used
# "headline topic: " but this says "write headline: ". The model still
# produces headlines (it generalises a little), but it's seeing an
# instruction it was never trained on — expect better headlines with
# "headline topic: ".
PREFIX = {"classify": "classify topic: ", "headline": "write headline: "}
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use bf16 for speed/memory if supported, else full fp32 (same T4 caveat as in training).
dtype = torch.bfloat16 if(device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# In[57]:


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 1) Load the frozen base model...
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,torch_dtype=dtype)
    # 2) ...then attach the LoRA adapter weights on top of it.
    model_lora = PeftModel.from_pretrained(model,model_path , adapter_name="model_lora")
    # .eval() switches OFF dropout so predictions are deterministic.
    model_lora.to(device).eval()
    return model_lora, tokenizer

# In[64]:


# @torch.no_grad(): we're only predicting, so skip gradient tracking
# (less memory, faster).
@torch.no_grad()
def classify(text,model, tokenizer):
    # Pick a label by asking: 'which label text is the model most likely to write?'
    #
    # Instead of letting the model generate freely (it might write "football"
    # instead of "sports"), we SCORE every allowed label and pick the best.
    # This guarantees the answer is always a valid class.
    #
    # Repeat the same article n times -> one row per candidate label, so all
    # labels are scored in a single forward pass.
    enc = tokenizer([PREFIX['classify'] + text]*n, return_tensors="pt",truncation=True, max_length=512).to(device)
    # Tokenize every label; pad so they form one rectangular tensor.
    tgt = tokenizer(lables, return_tensors="pt", padding=True).input_ids.to(device)
    # Replace pad tokens with -100 so they don't count in the loss.
    tgt = tgt.masked_fill(tgt == tokenizer.pad_token_id, -100)
    # Passing labels= makes the model predict each label token (teacher forcing).
    logits = model(**enc, labels=tgt).logits.float()
    # Per-token negative log-likelihood (how "surprised" the model is by each token).
    # cross_entropy wants (batch, vocab, seq), hence transpose(1, 2).
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none", ignore_index=-100)
    # Average log-likelihood per label (dividing by length so long labels like
    # "entertainment" aren't penalised for having more tokens).
    score = -nll.sum(1) / (tgt != -100).sum(1)
    # Softmax over labels -> numbers that sum to 1. Treat this as a RELATIVE
    # confidence, not a calibrated probability.
    probs = score.softmax(0)
    i = int(probs.argmax())
    return lables[i], float(probs[i])

@torch.no_grad()
def headline(text, model, tokenizer,num_beams=4, max_new_tokens=32):
    # Generate a headline with beam search.
    enc = tokenizer([PREFIX["headline"] + text], return_tensors="pt",
                   padding=True, truncation=True, max_length=512).to(device)
    # num_beams=4: keep the 4 best partial headlines at each step instead of
    #   greedily taking the single best token -> more fluent output.
    # no_repeat_ngram_size=3: forbid repeating any 3-word phrase (stops loops
    #   like "impeach ... afta dem impeach").
    # early_stopping=True: stop once all beams have finished.
    out = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3, early_stopping=True)
    # Token IDs -> text. Returns a list (one string per input).
    return tokenizer.batch_decode(out, skip_special_tokens=True)

# In[59]:


# Load base model + adapter onto the GPU.
model, tokenizer = load_model()

# In[66]:


# Qualitative check: run both tasks on 10 random test articles and read them.
# Things to look for:
#   - Is the label plausible? (Sports articles score ~0.98; the Igbo political
#     feud labelled "business" at 0.78 shows where the model is less sure.)
#   - Is the headline in the SAME language as the article?
#   - Repetitions or made-up names? (e.g. "Andre Suarez" in the Ghana example.)
# Next step: score the validation set properly — accuracy / macro-F1 for
# classification and ROUGE for headlines — so changes can be compared with numbers.
for i, row in df_test.sample(10).iterrows():
    print('For given text: ',row['text'])
    label, conf = classify(row['text'],model, tokenizer)
    print('label= ',label, ', conf=',conf)
    headline_gen = headline(row['text'],model, tokenizer)
    print('headline=',headline_gen)

# In[ ]:


# Leftover draft of an evaluate() method — fully commented out, so it never runs.
# (If you revive it: the device string should be "cuda", not "gpu", and the
#  check is torch.cuda.is_bf16_supported(), and the dtype is torch.bfloat16.)
# @classmethod
#     def evaluate(self,out_location):
#         dev = "gpu" if torch.cuda.is_available() else "cpu"
#         bf16 = dev=='gpu' and torch.cuda.bf16.is_available()
#         torchtype = torch.bffloat16 if bf16 else torch.float32

#         # model.to(dev).eval()

# ## Things to try next
# - **Fix the headline prefix** to `headline topic: ` and compare outputs.
# - **Measure, don't just eyeball:** accuracy / macro-F1 on `val.jsonl` for topics, ROUGE-L for headlines.
# - **Keep all test ids** for submission: remove leaked articles from *train* rather than *test*.
# - **On T4 GPUs**, try `bf16=False` (fp32): T4 only emulates bf16, so fp32 is often faster there.
# - **Per-language analysis:** check scores separately for `hau`, `yor`, `ibo`, `pcm` — the smallest language (Pidgin) may lag.
# - **Batch inference:** classify/generate many articles per forward pass to speed up the full test set.
