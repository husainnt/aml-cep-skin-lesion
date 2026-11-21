import pandas as pd, numpy as np, os
from sklearn.metrics import accuracy_score, f1_score, recall_score

HIGH_CONCERN = ["mel", "bcc", "akiec"]

def evaluate(name, y_true, pred):
    # here i put all the measures in one function so every model uses the same code
    y_true = pd.Series(y_true).reset_index(drop=True)
    pred = pd.Series(pred).reset_index(drop=True)

    # here i find which lesions are referred and which are really high concern
    referred = pred.isin(HIGH_CONCERN)
    is_high = y_true.isin(HIGH_CONCERN)

    result = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, pred), 3),
        "macro_f1": round(f1_score(y_true, pred, average="macro"), 3),
        "mel_recall": round(recall_score(y_true == "mel", pred == "mel"), 3),
       "referral_rate": round(float(referred.mean()), 3),
       "referral_sensitivity": round(float(referred[is_high].mean()), 3),
    }
    print(result)
    return result

def save_result(result, path=r"D:\ML CEP\CEP_CODE\results.csv"):
    # here i add the result as a row in results.csv
    # if the model name is already in the file i replace that row
    row = pd.DataFrame([result])
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old["model"] != result["model"]]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(path, index=False)

def to_lesion_level(proba, lesion_ids, y, classes):
    # here i average the 7 probabilities of all images that belong to the same lesion
    df = pd.DataFrame(proba, columns=classes)
    df["lesion_id"] = np.asarray(lesion_ids)
    mean_proba = df.groupby("lesion_id")[classes].mean()

    # here i get the true label of each lesion (we checked every lesion has only one dx)
    labels = pd.DataFrame({"lesion_id": np.asarray(lesion_ids), "dx": np.asarray(y)})
    labels = labels.drop_duplicates("lesion_id").set_index("lesion_id")["dx"]
    labels = labels.loc[mean_proba.index]

    # here i take the class with the highest averaged probability as the lesion prediction
    pred = mean_proba.idxmax(axis=1)
    return mean_proba, pred, labels