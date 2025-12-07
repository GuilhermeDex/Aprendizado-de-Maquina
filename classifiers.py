import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BitsAndBytesConfig, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from xgboost import XGBClassifier
def load_welfake(csv_path: str):
    df = pd.read_csv(csv_path)

    if 'Text' in df.columns:
        df['full_text'] = df['Text'].fillna('').astype(str)
    elif 'text' in df.columns:
        df['full_text'] = df['text'].fillna('').astype(str)
    else:
        candidate_cols = [c for c in df.columns
                          if ('text' in c.lower() or 'content' in c.lower()) and 'title' not in c.lower()]
        if len(candidate_cols) > 0:
            df['full_text'] = df[candidate_cols[0]].fillna('').astype(str)
        else:
            raise ValueError("Não encontrei coluna de texto (ex.: 'Text', 'text' ou 'content'). Ajuste o CSV.")

    if 'Label' in df.columns:
        df['label'] = df['Label'].astype(int)
    elif 'label' in df.columns:
        df['label'] = df['label'].astype(int)
    else:
        raise ValueError("Não encontrei coluna de rótulo (Label/label).")

    df = df[['full_text', 'label']].dropna().reset_index(drop=True)
    return df

# -------------------------
# Embedding extraction
# -------------------------
def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float() 
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()

def get_embeddings_for_texts(texts, tokenizer, model, device='cpu', batch_size=16, max_length=512):
    model.eval()
    embeddings = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = outputs.last_hidden_state
            batch_emb = mean_pooling(last_hidden, attention_mask)
            embeddings.append(batch_emb)
    embeddings = np.vstack(embeddings)
    return embeddings  # shape (len(texts), hidden_size)

# -------------------------
# Main: 5-fold experiments
# -------------------------
def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }
    metrics['roc_auc'] = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    return metrics


def mean_pooling(last_hidden, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask_expanded).sum(1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


def get_embeddings_for_texts(texts, tokenizer, model, device='cpu', batch_size=16, max_length=512):
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding texts'):
        batch_texts = [str(t) if not isinstance(t, str) else t for t in texts[i:i+batch_size]]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            last_hidden = model(**enc).last_hidden_state
            emb = mean_pooling(last_hidden, enc['attention_mask']).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)


# -------------------------
# Experimentos separados
# -------------------------
def run_cv_experiment(
    texts,
    labels,
    model_name=None,
    mode="svm",  # "svm", "roberta", "llama", "xgboost", "naive bayes
    emb_model_name="sentence-transformers/all-mpnet-base-v2",
    output_root="outputs",
    device=None,
    n_splits=5,
    random_state=42
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
    os.makedirs(output_root, exist_ok=True)

    texts = [str(t) if not isinstance(t, str) else t for t in texts]
    labels = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results, preds_all = [], []

    # ========== (1) SVM ==========
    if mode == "svm":
        print(f"Extraindo embeddings com {emb_model_name}...")
        tok = AutoTokenizer.from_pretrained(emb_model_name)
        model = AutoModel.from_pretrained(emb_model_name).to(device)
        emb = get_embeddings_for_texts(texts, tok, model, device=device)
        del model
        torch.cuda.empty_cache()

        for fold_idx, (train_idx, test_idx) in tqdm(enumerate(skf.split(emb, labels), 1), desc='SVM PROGRESS', total=n_splits):
            print(f"\n=== Fold {fold_idx}/{n_splits} ===")
            Xtr, Xte = emb[train_idx], emb[test_idx]
            ytr, yte = labels[train_idx], labels[test_idx]
            clf = svm.SVC(kernel='linear', C=1.0, probability=True, random_state=random_state)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]
            mets = compute_metrics(yte, pred, proba)
            results.append(mets)
            preds_all.append(pd.DataFrame({
                'text': [texts[i] for i in test_idx],
                'true_label': yte,
                'pred_label': pred,
                'pred_proba': proba,
                'fold': fold_idx
            }))
        del emb
        torch.cuda.empty_cache()

    # ========== (X) XGBOOST ==========
    elif mode == "xgboost":
        print(f"Extraindo embeddings com {emb_model_name} para XGBoost...")
        tok = AutoTokenizer.from_pretrained(emb_model_name)
        model = AutoModel.from_pretrained(emb_model_name).to(device)
        emb = get_embeddings_for_texts(texts, tok, model, device=device)
        del model
        torch.cuda.empty_cache()

        tree_method = 'gpu_hist' if (torch.cuda.is_available()) else 'hist'
        for fold_idx, (train_idx, test_idx) in tqdm(enumerate(skf.split(emb, labels), 1), desc='XGB PROGRESS', total=n_splits):
            print(f"\n=== Fold {fold_idx}/{n_splits} ===")
            Xtr, Xte = emb[train_idx], emb[test_idx]
            ytr, yte = labels[train_idx], labels[test_idx]

            clf = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method=tree_method
            )
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]
            mets = compute_metrics(yte, pred, proba)
            results.append(mets)
            preds_all.append(pd.DataFrame({
                'text': [texts[i] for i in test_idx],
                'true_label': yte,
                'pred_label': pred,
                'pred_proba': proba,
                'fold': fold_idx
            }))
        del emb
        torch.cuda.empty_cache()
        
    elif mode == "naive_bayes":
        print(f"Extraindo embeddings com {emb_model_name} para Naive Bayes...")
        tok = AutoTokenizer.from_pretrained(emb_model_name)
        model = AutoModel.from_pretrained(emb_model_name).to(device)

        emb = get_embeddings_for_texts(texts, tok, model, device=device)
        del model
        torch.cuda.empty_cache()

        for fold_idx, (train_idx, test_idx) in tqdm(
            enumerate(skf.split(emb, labels), 1),
            desc='NB PROGRESS',
            total=n_splits
        ):
            print(f"\n=== Fold {fold_idx}/{n_splits} ===")
            Xtr, Xte = emb[train_idx], emb[test_idx]
            ytr, yte = labels[train_idx], labels[test_idx]

            clf = GaussianNB()
            clf.fit(Xtr, ytr)

            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte)[:, 1]

            mets = compute_metrics(yte, pred, proba)
            results.append(mets)

            preds_all.append(pd.DataFrame({
                'text': [texts[i] for i in test_idx],
                'true_label': yte,
                'pred_label': pred,
                'pred_proba': proba,
                'fold': fold_idx
            }))

        del emb
        torch.cuda.empty_cache()
        
    # ========== (2) RoBERTa / LLaMA ==========
    elif mode in ["roberta", "llama"]:
        print(f"Carregando modelo {mode.upper()} com quantização 8 bits...")
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        if mode == "roberta":
            tok = AutoTokenizer.from_pretrained(model_name)
            config = RobertaConfig.from_pretrained("roberta-base", num_labels=2)
            base_model = RobertaModel.from_pretrained(
                "roberta-base",
                device_map="auto"
            )
            model = RobertaForSequenceClassification(config)
            model.roberta = base_model
            model.to(device)
        elif mode == "llama":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 # ou torch.float16
            )
            tok = AutoTokenizer.from_pretrained(model_name, max_lenth = 128, padding=True, Truncation= True, token="TOKEN")
            tok.pad_token = tok.eos_token  # Reuse EOS as padding
            model = AutoModelForSequenceClassification.from_pretrained(model_name, token="TOKEN", device_map = "auto", quantization_config=quant_cfg)

        batch_size = 1

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels), 1):
            if fold_idx == 3: break
            print(f"\n=== Fold {fold_idx}/{n_splits} ===")
            
            y_test = labels[test_idx]
            test_texts = [texts[i] for i in test_idx]
        
            all_pred = []
            all_proba = []
        
            for i in tqdm(range(0, len(test_texts), batch_size), desc="Progress..."):
                batch_texts = test_texts[i:i + batch_size]
        
                # Tokenize batch
                enc = tok(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
                
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
        
                model.eval()
                with torch.inference_mode():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    proba = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    pred = (proba >= 0.5).astype(int)
        
              
                all_proba.extend(proba)
                all_pred.extend(pred)
        
               
                del enc, logits, input_ids, attention_mask
                torch.cuda.empty_cache()
        
            all_pred = np.array(all_pred)
            all_proba = np.array(all_proba)
        
            mets = compute_metrics(y_test, all_pred, all_proba)
            results.append(mets)
        
            preds_all.append(pd.DataFrame({
                'text': test_texts,
                'true_label': y_test,
                'pred_label': all_pred,
                'pred_proba': all_proba,
                'fold': fold_idx
            }))

        del model
        torch.cuda.empty_cache()

    # ========== Salvar ==========
    if len(preds_all) == 0:
        print("Nenhuma predição foi gerada (preds_all vazio). Verifique o modo escolhido.")
        return {}

    df_preds = pd.concat(preds_all, ignore_index=True)
    df_preds.to_csv(os.path.join(output_root, f"preds_{mode}.csv"), index=False)
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(os.path.join(output_root, f"metrics_{mode}.csv"), index=False)

    print("\n=== Resultados médios ===")
    print(df_metrics.mean())
    return df_metrics.mean().to_dict()

def prep_data(df: pd.DataFrame):

    aux = pd.read_csv("/kaggle/input/data-welfake/WELFake_Dataset.csv")

    df['info'] = df.apply(lambda row: f"Linguistic Characteristics: a_n_ent_pw: {row['a_n_ent_pw']}, simp_adj_var: {row['simp_adj_var']}, simp_ttr: {row['simp_ttr']}, n_adj: {row['n_adj']}, a_adj_pw: {row['a_adj_pw']}, fkre: {row['fkre']}, rt_fast: {row['rt_fast']}|Text: " + str(row['text']), axis=1)

    texts = df['info'].tolist()
    labels = aux['label'].astype(int).to_numpy()

    del(aux)
    return texts, labels


# -------------------------
# Execução por etapas
# -------------------------
if __name__ == "__main__":
    csv_path = "/kaggle/input/data-welfake/WELFake_Dataset.csv"
    df = pd.read_csv(csv_path)
    texts, labels = df['text'], df['label']
    print("\n=== Rodando Naive Bayes===")
    run_cv_experiment(texts, labels, mode="naive_bayes")
