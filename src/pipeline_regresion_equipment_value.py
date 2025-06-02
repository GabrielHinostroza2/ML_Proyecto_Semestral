# %% [markdown]
# # Pipeline CS:GO – Regresión RoundStartingEquipmentValue (ventana 0–30 s)

# %%
import os, warnings
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mstats, spearmanr

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LassoCV, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost no instalado, lo omito.")

RANDOM_STATE = 42
TARGET       = "RoundStartingEquipmentValue"
TIME_WINDOW  = 30
RAW_PATH     = "Anexo_ET_demo_round_traces_2022.csv"

# %% ------------- 1. Carga y limpieza ----------------------------------------
df = pd.read_csv(RAW_PATH, sep=";", low_memory=False)
drop_cols = ["Unnamed: 0","InternalTeamId","FirstKillTime","Tick","EventTime"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

for tcol in ["TimeAlive","TravelledDistance"]:
    df[tcol] = (df[tcol].astype(str)
                        .str.replace(r"\.","",regex=True)
                        .pipe(pd.to_numeric, errors="coerce")
                        .div(1_000_000))

for cat in ["Map","Team","TeamSide"]:
    df[cat] = df[cat].astype("category")
for b in ["RoundWinner","MatchWinner","Survived","AbnormalMatch"]:
    df[b] = df[b].astype("bool")

combat = ["RoundKills","RoundAssists","RoundHeadshots","RoundFlankKills",
          "MatchKills","MatchAssists","MatchHeadshots","MatchFlankKills"]
df[combat] = df[combat].fillna(0).astype(int)
for col in df.select_dtypes(include=["float64","int64"]).columns:
    if col not in combat and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

df = df[df["TimeAlive"].fillna(0) <= TIME_WINDOW]
if "AbnormalMatch" in df.columns:
    df = df[~df["AbnormalMatch"]]
df = df.dropna().drop_duplicates()
print("Post-limpieza", df.shape)

# %% ------------- 2. Feature engineering -------------------------------------
df["IsPistolRound"] = df["RoundStartingEquipmentValue"] < 1500
df["HeadshotRatio"] = np.where(df["RoundKills"]>0, df["RoundHeadshots"] / df["RoundKills"], 0)
df["KillEfficiency"] = df["RoundKills"] / (df["MatchKills"] + 1)
df["AssistRatio"] = df["RoundAssists"] / (df["MatchAssists"] + 1)

for col in df.select_dtypes(include=["int64","float64"]).columns:
    df[col] = mstats.winsorize(df[col], limits=(0.05,0.05))

print("Con FE", df.shape)

# %% ------------- 3. Split por MatchId ---------------------------------------
ID_COLS = ["MatchId","RoundId"]
X_full  = df.drop(columns=ID_COLS + [TARGET])
y_full  = df[TARGET]
groups  = df["MatchId"]

splitter = GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=RANDOM_STATE)
tr_idx, te_idx = next(splitter.split(X_full,y_full,groups))

cat_cols  = X_full.select_dtypes(exclude=["number"]).columns
X_encoded = pd.get_dummies(X_full, columns=cat_cols, drop_first=True)

X_train, X_test = X_encoded.iloc[tr_idx], X_encoded.iloc[te_idx]
y_train, y_test = y_full.iloc[tr_idx],  y_full.iloc[te_idx]

X_train_cb, X_test_cb = X_full.iloc[tr_idx], X_full.iloc[te_idx]

# %% ------------- 4. Top-25 features (numéricas) -----------------------------
def top_features(X,y,k=25):
    mi = mutual_info_regression(X,y,random_state=RANDOM_STATE)
    f  = f_regression(X,y)[0]
    lasso = LassoCV(cv=5,random_state=RANDOM_STATE).fit(X,y).coef_
    score = np.abs(mi)+np.abs(f)+np.abs(lasso)
    return X.columns[np.argsort(score)[-k:]]

TOP = top_features(X_train, y_train, 25)
X_train_f, X_test_f = X_train[TOP], X_test[TOP]

# %% ------------- 5. Modelos --------------------------------------------------
models = {
    "Poisson": PoissonRegressor(alpha=1e-4, max_iter=1000),
    "RandForest": RandomForestRegressor(
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradBoost": GradientBoostingRegressor(
        n_estimators=600, learning_rate=0.05, random_state=RANDOM_STATE
    ),
}
if CATBOOST_AVAILABLE:
    models["CatBoost"] = CatBoostRegressor(
        depth=6, learning_rate=0.05, iterations=700,
        random_state=RANDOM_STATE, verbose=False
    )

def evaluate(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    spearman = spearmanr(yte, pred).correlation
    if np.isnan(spearman):
        spearman = None
    return {
        "MAE": mean_absolute_error(yte, pred),
        "RMSE": mean_squared_error(yte, pred, squared=False),
        "R2": r2_score(yte, pred),
        "Spearman": spearman
    }, pred

# %% ------------- 6. Entrenamiento y métricas --------------------------------
records, preds_dict = [], {}
for name, mdl in models.items():
    print("→", name)
    if name=="CatBoost":
        metrics, pred = evaluate(mdl, X_train_cb, y_train, X_test_cb, y_test)
    else:
        metrics, pred = evaluate(mdl, X_train_f, y_train, X_test_f, y_test)
    metrics["Modelo"]=name
    records.append(metrics)
    preds_dict[name]=pred

results = pd.DataFrame(records).round(3).sort_values("MAE").reset_index(drop=True)
print("\nMétricas:\n",results)

# %% ------------- 7. Diagnóstico rápido --------------------------------------
best_name = results.iloc[0]["Modelo"]; preds = preds_dict[best_name]

print("\nVarianza y_test :", round(y_test.var(),4))
print("Moda y_test     :", y_test.mode()[0])
print("MAE baseline    :", round(mean_absolute_error(y_test, np.full_like(y_test, y_test.mode()[0])),3))

# %% ------------- 8. Visualización best model --------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.3)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"--")
plt.title(f"Parity – {best_name}")
plt.xlabel("Real"); plt.ylabel("Predicho")
plt.show()

plt.hist(y_test - preds, bins=20)
plt.title(f"Errores – {best_name}")
plt.show()

# %% ------------- 9. Guardar modelo ------------------------------------------
import joblib, pathlib
pathlib.Path("models").mkdir(exist_ok=True)
MODEL_PATH = f"models/{best_name}_equipmentvalue.pkl"
if best_name=="CatBoost":
    models[best_name].fit(X_train_cb, y_train)
    joblib.dump(models[best_name], MODEL_PATH)
else:
    models[best_name].fit(X_train_f, y_train)
    joblib.dump(models[best_name], MODEL_PATH)
print("✓ Modelo guardado en", MODEL_PATH)
