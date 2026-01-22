import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import json
import datetime

# PC алгоритм
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
BOOTSTRAP_N = 300
OUTPUT_DIR = "./output"


def ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- 1) Load data & missing counts ----------
def load_data(path):
    df = pd.read_excel(path)
    return df


def missing_summary(df):
    return df.isna().sum()


# ---------- 2) Simple RF-imputer per-column ----------
def rf_impute_column(df, target, features=None, n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE):
    df_loc = df.copy()
    if features is None:
        features = [c for c in df.columns if c != target]

    train_idx = df_loc[df_loc[target].notna() & df_loc[features].notna().all(axis=1)].index
    predict_idx = df_loc[df_loc[target].isna() & df_loc[features].notna().all(axis=1)].index

    if len(predict_idx) == 0:
        return df_loc, 0

    if len(train_idx) < 10:
        df_loc.loc[predict_idx, target] = df_loc[target].median()
        return df_loc, int(len(predict_idx))

    X_train = df_loc.loc[train_idx, features].copy()
    y_train = df_loc.loc[train_idx, target].copy()
    X_pred = df_loc.loc[predict_idx, features].copy()

    if 'DeviceID' in X_train.columns:
        X_train['DeviceID'] = X_train['DeviceID'].astype(int)
        X_pred['DeviceID'] = X_pred['DeviceID'].astype(int)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_pred)
    df_loc.loc[predict_idx, target] = preds
    return df_loc, int(len(predict_idx))


# ---------- DAG drawing with PC ----------
def draw_dag_pc(df, path_out, alpha=0.05):
    df_numeric = df.select_dtypes(include=[np.number])
    X = df_numeric.to_numpy()
    col_names = list(df_numeric.columns)

    # Запуск PC алгоритму
    cg = pc(X, alpha=alpha, indep_test_fn=fisherz)
    
    # Візуалізація через networkx
    import networkx as nx
    G = nx.DiGraph()
    for i, col in enumerate(col_names):
        G.add_node(col)
    
    # cg.G.graph — це матриця суміжності (adj_matrix)
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            if cg.G.graph[i, j] != 0:  # якщо є напрямлене ребро
                G.add_edge(col_names[i], col_names[j])
    
    plt.figure(figsize=(10,6))
    pos = nx.spring_layout(G, seed=RANDOM_STATE)
    nx.draw(G, with_labels=True, node_color="lightblue", arrows=True)
    plt.title("DAG побудований за PC")
    plt.tight_layout()
    plt.savefig(path_out)
    plt.close()
    return True, path_out


# ----------  DML estimation ----------
def dml_ate(df, treatment='WatchTime', outcome='InteractionQuality', confounders=None, n_boot=BOOTSTRAP_N):
    if confounders is None:
        confounders = ['SessionQuality','Recommendations']

    X = df[confounders]
    W = df[treatment]
    Y = df[outcome]

    # Крок 1: Навчання ML моделей
    model_y = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model_t = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE)
    model_y.fit(X, Y)
    model_t.fit(X, W)

    # Крок 2: Отримання залишків
    Y_res = Y - model_y.predict(X)
    X_res = W - model_t.predict(X)

    # Крок 3: Оцінити причинний ефект
    theta = np.sum(X_res * Y_res) / np.sum(X_res**2)

    rng = np.random.RandomState(RANDOM_STATE)
    boot_thetas = []
    for _ in range(n_boot):
        idx = rng.choice(len(X_res), size=len(X_res), replace=True)
        theta_b = np.sum(X_res[idx] * Y_res[idx]) / np.sum(X_res[idx]**2)
        boot_thetas.append(theta_b)
    ci_low, ci_high = np.percentile(boot_thetas, [2.5, 97.5])

    return theta, (ci_low, ci_high)


def save_histograms(df_original, df_imputed, out_dir):
    plt.figure(figsize=(8,4))
    plt.hist(df_original['WatchTime'].dropna(), bins=30, alpha=0.6, label='original (non-missing)')
    plt.hist(df_imputed['WatchTime'], bins=30, alpha=0.4, label='imputed (full)')
    plt.legend(); plt.title('WatchTime: original vs imputed'); plt.tight_layout()
    p1 = os.path.join(out_dir, 'watchtime_dist.png')
    plt.savefig(p1); plt.close()

    plt.figure(figsize=(8,4))
    plt.hist(df_original['SessionQuality'].dropna(), bins=30, alpha=0.6, label='original (non-missing)')
    plt.hist(df_imputed['SessionQuality'], bins=30, alpha=0.4, label='imputed (full)')
    plt.legend(); plt.title('SessionQuality: original vs imputed'); plt.tight_layout()
    p2 = os.path.join(out_dir, 'sessionquality_dist.png')
    plt.savefig(p2); plt.close()
    return p1, p2


def write_report(report_path, summary_dict):
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("#Звіт\n\n")
        f.write(f"Дата створення: {datetime.datetime.now().isoformat()}\n\n")
        f.write("## Коротка інформація\n\n")
        f.write(json.dumps(summary_dict, indent=2, ensure_ascii=False))


def main(path):
    ensure_output()

    # Load
    df = load_data(path)
    missing_before = missing_summary(df)
    print("Missing before:\n", missing_before)

    # Imputation
    features_sq = ['DeviceID','BatteryWear','Recommendations','Click','InteractionQuality']
    df_step, n_imp_sq = rf_impute_column(df, 'SessionQuality', features=features_sq, n_estimators=RF_N_ESTIMATORS)
    features_wt = ['DeviceID','BatteryWear','SessionQuality','Recommendations','Click','InteractionQuality']
    df_imputed, n_imp_wt = rf_impute_column(df_step, 'WatchTime', features=features_wt, n_estimators=RF_N_ESTIMATORS)
    missing_after = missing_summary(df_imputed)
    print("Missing after:\n", missing_after)
    imputed_path = os.path.join(OUTPUT_DIR, "data_imputed.csv")
    df_imputed.to_csv(imputed_path, index=False)

    # DAG using PC
    dag_path = os.path.join(OUTPUT_DIR, "dag_pc.png")
    dag_created, dag_file = draw_dag_pc(df_imputed, dag_path)

    # DML 
    confounders = ['BatteryWear','SessionQuality','Recommendations']
    dml_theta, dml_ci = dml_ate(df_imputed, treatment='WatchTime', outcome='InteractionQuality', confounders=confounders, n_boot=BOOTSTRAP_N)
    print("DML ATE:", dml_theta)
    print("95% CI:", dml_ci)

    p_watch, p_session = save_histograms(df, df_imputed, OUTPUT_DIR)

    summary = {
        'missing_before': missing_before.to_dict(),
        'missing_after': missing_after.to_dict(),
        'n_imputed_SessionQuality': n_imp_sq,
        'n_imputed_WatchTime': n_imp_wt,
        'imputed_file': imputed_path,
        'dag_created': dag_created,
        'dag_file': dag_file if dag_created else None,
        'dml_ate_watchtime': float(dml_theta),
        'dml_ate_ci': [float(dml_ci[0]), float(dml_ci[1])],
        'plots': [p_watch, p_session]
    }
    report_md = os.path.join(OUTPUT_DIR, "report.md")
    write_report(report_md, summary)

    print("All done. Output files in:", OUTPUT_DIR)
    return summary


if __name__ == "__main__":
    path = "/Users/darynanagaevskaya/Desktop/4year/AI/dataset.xlsx"
    s = main(path)
    print(json.dumps(s, indent=2, ensure_ascii=False))
